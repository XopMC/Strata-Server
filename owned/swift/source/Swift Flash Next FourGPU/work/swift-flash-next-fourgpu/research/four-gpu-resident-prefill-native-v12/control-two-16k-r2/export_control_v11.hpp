#pragma once
#include "reference_v11.hpp"
#include <optional>
namespace four_reference_v11::control {
struct State{Metadata m;std::array<Bank,2>banks;std::string prefix;bool done=false;};
inline std::optional<State>&state(){static std::optional<State>s;return s;}
template<class O>void startup(const O&o,const std::array<Bank,2>&banks,std::span<const int32_t>R,
 const std::vector<std::pair<int32_t,int32_t>>&rank,const std::array<uint64_t,2>&pp_prices){
 const char*p=std::getenv("STRATA_FOUR_CONTROL_EXPORT_V11");need(p&&*p=='/',"absolute new independent control export prefix");need(!state(),"control export once");
 need(o.adapt_swaps==0&&(o.prefill_chunk==16384||o.prefill_chunk==8192)&&!o.prefill_auto,"control exact fixed profile required");Metadata m;m.pp_price=pp_prices;m.rows=uint32_t(o.prefill_chunk);m.options=tuple(o);m.map.assign(R.begin(),R.end());for(auto[l,e]:rank){need(l>=0&&l<48&&e>=0&&e<512,"control rank bounds");m.rank.push_back(l*512+e);}for(int d=0;d<2;++d){auto*c=banks[d].cache;need(c&&c->valid()&&c->device_slot(0)!=nullptr,"reference must be independent REAL original cache");m.slots[d]=uint32_t(c->slots());m.bytes[d]=uint64_t(c->bytes());for(uint32_t q=0;q<=m.slots[d];++q)m.offsets[d].push_back(c->slot_offset(q));}
 (void)Metadata::decode(m.encode());checked_tables(banks,R,m);write_new(std::string(p)+".startup",m);receipt(m,"independent-startup");state()=State{std::move(m),banks,p,false};
}
inline void checked_postprefill(std::span<const int64_t>ids,std::span<const int32_t>R){need(state()&&!state()->done,"independent control postprefill once");auto&s=*state();checked_tables(s.banks,R,s.m);s.m.input_count=ids.size();s.m.input_fnv=input_hash(ids);s.m.postprefill=true;write_new(s.prefix+".postprefill",s.m);receipt(s.m,"independent-postprefill");s.done=true;}
}
