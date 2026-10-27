#include "strata/ds41/native_checkpoint.hpp"
#include "strata/ds41/config.hpp"
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
namespace strata::ds41 {
namespace {
[[noreturn]]void bad(const std::string&why){throw std::invalid_argument("native checkpoint: "+why);}
}
void NativeCheckpoint::validate_geometry(const std::string&text){
    if(text.empty()||text.size()>65536)bad("missing/oversized geometry");
    const std::map<std::string,double> scalars={
        {"dim",kDim},{"n_layers",kLayers},{"vocab_size",kVocab},{"n_heads",kHeads},{"head_dim",kHeadDim},
        {"rope_head_dim",kRopeDim},{"q_lora_rank",kQLora},{"o_groups",kOGroups},{"o_lora_rank",kOLora},
        {"window_size",kWindow},{"n_routed_experts",kExperts},{"n_activated_experts",kTopK},{"moe_inter_dim",kMoeInter},
        {"route_scale",kRouteScale},{"swiglu_limit",kSwigluLimit},{"hc_mult",kHc},{"hc_sinkhorn_iters",kSinkhornIters},
        {"hc_eps",1e-6},{"norm_eps",1e-20},{"index_n_heads",kIndexHeads},{"index_head_dim",kIndexDim},
        {"index_topk",kIndexTopK},{"candidate_source_layer",kCandidateLayer},{"candidate_topk_blocks",kCandidateBlocks},
        {"candidate_block_size",kCandidateBlock},{"rope_theta",kRopeTheta},{"compress_rope_theta",kCompressRopeTheta},
        {"rope_factor",kRopeFactor},{"beta_fast",kBetaFast},{"beta_slow",kBetaSlow},{"original_seq_len",kOriginalSeqLen},
        {"n_shared_experts",1},{"n_mtp_layers",3},{"engram_n_heads",8},{"engram_head_dim",256},
        {"engram_max_ngram_size",4},{"engram_compressed_vocab_size",99092},{"engram_pad_id",2}};
    std::vector<int64_t> ratios(kCompressRatio.begin(),kCompressRatio.end());ratios.insert(ratios.end(),3,0);
    const std::map<std::string,std::vector<int64_t>> arrays={
        {"kv_source_layers",{2,8,14,20}},{"index_source_layers",{2,8,14,20,24,28,32,36}},
        {"engram_layer_ids",{1,14}},{"compress_ratios",ratios},{"engram_num_embeddings",{384006168,384016682}}};
    std::istringstream stream(text);std::string line;std::set<std::string>seen;
    while(std::getline(stream,line)){
        if(line.empty())continue;std::istringstream row(line);std::string key;row>>key;
        if(!seen.insert(key).second)bad("duplicate geometry key "+key);
        auto scalar=scalars.find(key);auto array=arrays.find(key);
        if(scalar!=scalars.end()){
            double value;std::string extra;
            if(!(row>>value)||!std::isfinite(value)||value!=scalar->second||(row>>extra))bad("geometry differs for "+key);
        }else if(array!=arrays.end()){
            std::vector<int64_t>values;int64_t value;while(row>>value)values.push_back(value);
            if(!row.eof()||values!=array->second)bad("geometry differs for "+key);
        }else if(key=="score_func"){
            std::string value,extra;if(!(row>>value)||value!="sqrtsoftplus"||(row>>extra))bad("unsupported router score function");
        }else bad("unknown geometry key "+key);
    }
    if(seen.size()!=scalars.size()+arrays.size()+1)bad("incomplete geometry");
}
NativeCheckpoint::NativeCheckpoint(std::shared_ptr<const BorrowedWeights>w,const std::string&geometry):weights_(std::move(w)){
    validate_geometry(geometry);if(!weights_)bad("missing borrowed allocation owner");
    rows_=std::make_shared<BorrowedRows>(weights_);
    auto dense=[&](const std::string&name,TensorEncoding encoding,std::vector<int64_t>shape){
        const auto&t=weights_->at(name);
        if(t.encoding!=encoding||t.shape!=shape)bad("dense tensor encoding/shape differs: "+name);
        dense_bytes_=(dense_bytes_+255)&~size_t(255);
        if(t.bytes>(24ull<<30)-dense_bytes_)bad("dense arena exceeds24GiB");dense_bytes_+=t.bytes;dense_.push_back(name);
    };
    auto fp8=[&](const std::string&prefix,int64_t rows,int64_t columns){
        dense(prefix+".weight",TensorEncoding::F8e4m3,{rows,columns});
        dense(prefix+".scale",TensorEncoding::E8m0,{(rows+31)/32,(columns+31)/32});
    };
    dense("embed.weight",TensorEncoding::Bf16,{kVocab,kDim});dense("head.weight",TensorEncoding::Bf16,{kVocab,kDim});
    dense("norm.weight",TensorEncoding::Bf16,{kDim});
    for(int l=0;l<kLayers;l++){
        const auto p="layers."+std::to_string(l)+".";
        fp8(p+"attn.wq_a",kQLora,kDim);fp8(p+"attn.wq_b",kHeads*kHeadDim,kQLora);
        fp8(p+"attn.wkv",kHeadDim,kDim);fp8(p+"attn.wo_a",kOGroups*kOLora,kHeads*kHeadDim/kOGroups);fp8(p+"attn.wo_b",kDim,kOGroups*kOLora);
        dense(p+"attn.q_norm.weight",TensorEncoding::Bf16,{kQLora});dense(p+"attn.kv_norm.weight",TensorEncoding::Bf16,{kHeadDim});
        dense(p+"attn.attn_sink",TensorEncoding::F32,{kHeads});
        for(auto which:{"attn","ffn"}){
            dense(p+which+"_norm.weight",TensorEncoding::Bf16,{kDim});
            dense(p+"hc_"+which+"_fn",TensorEncoding::F32,{kHcMix,kHc*kDim});
            dense(p+"hc_"+which+"_base",TensorEncoding::F32,{kHcMix});dense(p+"hc_"+which+"_scale",TensorEncoding::F32,{3});
        }
        dense(p+"ffn.gate.weight",TensorEncoding::Bf16,{kExperts,kDim});
        dense(p+"ffn.gate.bias",TensorEncoding::F32,{kExperts});dense(p+"ffn.gate.bias_vl",TensorEncoding::F32,{kExperts});
        fp8(p+"ffn.shared_experts.w1",kMoeInter,kDim);fp8(p+"ffn.shared_experts.w3",kMoeInter,kDim);fp8(p+"ffn.shared_experts.w2",kDim,kMoeInter);
        if(is_kv_source(l)){
            dense(p+"attn.compressor.wkv.weight",TensorEncoding::Bf16,{kHeadDim,kDim});
            dense(p+"attn.compressor.norm.weight",TensorEncoding::Bf16,{kHeadDim});
            if(kCompressRatio[l]>1)dense(p+"attn.compressor.wgate.weight",TensorEncoding::Bf16,{kHeadDim,kDim});
            dense(p+"attn.indexer.wk.weight",TensorEncoding::Bf16,{kIndexDim,kHeadDim});dense(p+"attn.indexer.k_norm.weight",TensorEncoding::Bf16,{kIndexDim});
        }
        if(is_index_source(l)){
            fp8(p+"attn.indexer.wq_b",kIndexHeads*kIndexDim,kQLora);dense(p+"attn.indexer.weights_proj.weight",TensorEncoding::Bf16,{kIndexHeads,kDim});
        }
        if(is_engram_layer(l)){
            fp8(p+"engram.wkv",(kHc+1)*kDim,24*256);
            dense(p+"engram.q_weight",TensorEncoding::Bf16,{kHc,kDim});dense(p+"engram.k_weight",TensorEncoding::Bf16,{kHc,kDim});
            const auto&table=weights_->at(p+"engram.embed.weight");
            const int64_t n=l==1?384006168:384016682;
            if(table.shape!=std::vector<int64_t>{n,256})bad("Engram row geometry differs");
            if(table.encoding==TensorEncoding::F8e4m3){
                const auto&s=weights_->at(p+"engram.embed.scale");
                if(s.encoding!=TensorEncoding::E8m0||s.shape!=std::vector<int64_t>{n,8})bad("Engram native scale geometry differs");
            }else if(table.encoding!=TensorEncoding::Uniform)bad("unsupported Engram encoding");
        }
        for(int e=0;e<kExperts;e++)for(auto which:{"w1","w3","w2"}){
            bool down=std::string(which)=="w2";
            weights_->inspect_projection(p+"ffn.experts."+std::to_string(e)+"."+which,down?kDim:kMoeInter,down?kMoeInter:kDim);
        }
    }
}
}
