#include "strata/ds41/native_gpu_experts_compact_v252.hpp"
#include <cstdio>
#include "strata/ds41/native_fp8_output16_v244.h"
#include <atomic>
#include "strata/ds41/native_expert_gemv.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <set>
namespace strata::ds41 {
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("native GPU experts: ")+cudaGetErrorString(e));}
// The backbone remains on GPU0; cached FFNs can run on one explicitly selected
// visible device. Each call restores its caller's device, including on failure.
struct ExpertDeviceScope{
 int previous=0;
 explicit ExpertDeviceScope(int device){check(cudaGetDevice(&previous));if(device!=previous)check(cudaSetDevice(device));}
 ~ExpertDeviceScope(){cudaSetDevice(previous);}
};
}
void NativeGpuExperts::load(int layer,int expert,int preferred){
 std::array<StagedProjection,3>source;std::array<CompactProjectionV248,3>compact{};
 std::array<bool,3>blocked{};size_t bytes=0,max_compact=0;int at=0;
 for(const char*which:{"w1","w3","w2"}){
  const bool down=std::strcmp(which,"w2")==0;
  const std::string prefix="layers."+std::to_string(layer)+".ffn.experts."+std::to_string(expert)+"."+which;
  blocked[at]=weights_->is_output16_compact(prefix);
  if(blocked[at]){
   compact[at]=weights_->compact_projection(prefix,down?5120:2304,down?2304:5120);
   source[at].projection=weights_->inspect_projection(prefix,down?5120:2304,down?2304:5120);
   max_compact=std::max(max_compact,compact[at].bytes);
  }else source[at]=weights_->stage_projection(prefix,down?5120:2304,down?2304:5120);
  const auto&p=source[at++].projection;bytes=(bytes+255)&~size_t(255);bytes+=size_t(p.rows)*p.weight_row_bytes;
  if(p.scale_row_bytes){bytes=(bytes+255)&~size_t(255);bytes+=size_t(p.rows)*p.scale_row_bytes;}
 }
 int selected=-1;
 for(size_t i=0;i<arenas_.size();++i){int candidate=(preferred+int(i))%int(arenas_.size());if(bytes<=arenas_[candidate].budget-arenas_[candidate].used){
  ExpertDeviceScope scope(arenas_[candidate].device);size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
  const size_t aligned=(bytes+2*1024*1024-1)&~size_t(2*1024*1024-1);
  if(free>=aligned+(size_t(1)<<30)+workspace_bytes_+max_compact){selected=candidate;break;}
 }}
 if(selected<0)return;
 auto&arena=arenas_[selected];ExpertDeviceScope device_scope(arena.device);
 Entry entry;entry.bytes=bytes;entry.arena=selected;
 auto allocation=cudaMalloc(&entry.storage,bytes);
 if(allocation==cudaErrorMemoryAllocation){cudaGetLastError();arena.budget=arena.used;return;}
 check(allocation);
 uint8_t*compact_device=nullptr;
 try{
  if(max_compact)check(cudaMalloc(&compact_device,max_compact));
  size_t offset=0;int i=0;uint64_t compact_uploaded=0,inverse_bytes=0;
  for(auto&retained:source){const auto&p=retained.projection;
   if(blocked[i]){
    const auto&c=compact[i];
    auto reserve=[&](size_t n){offset=(offset+255)&~size_t(255);if(n>entry.bytes-offset)throw std::runtime_error("bounded compact cache inverse exceeds original entry");auto*dest=entry.storage+offset;offset+=n;return dest;};
    const size_t weight_bytes=size_t(p.rows)*p.weight_row_bytes,scale_bytes=size_t(p.rows)*p.scale_row_bytes;
    auto*w=reserve(weight_bytes);auto*scales=scale_bytes?reserve(scale_bytes):nullptr;
    check(cudaMemcpy(compact_device,c.data,c.bytes,cudaMemcpyHostToDevice));
    if(ds41_output16_to_rowpacked_gpu_v244(compact_device,c.bytes,c.bits,c.rows,c.columns,
        w,weight_bytes,scales,scale_bytes,0))throw std::runtime_error(ds41_fp8_output16_v244_error());
    check(cudaStreamSynchronize(0)); // one-time inverse; temporary may be reused/freed
    entry.projections[i++]={w,scales,p.bits};compact_uploaded+=c.bytes;inverse_bytes+=weight_bytes+scale_bytes;
   }else{
    auto copy=[&](const uint8_t*data,size_t n){offset=(offset+255)&~size_t(255);auto*dest=entry.storage+offset;offset+=n;check(cudaMemcpy(dest,data,n,cudaMemcpyHostToDevice));return dest;};
    entry.projections[i++]={copy(p.weights,size_t(p.rows)*p.weight_row_bytes),p.scales?copy(p.scales,size_t(p.rows)*p.scale_row_bytes):nullptr,p.bits};
   }
  }
  if(offset!=entry.bytes)throw std::runtime_error("compact cache entry original byte accounting differs");
  if(compact_device){check(cudaFree(compact_device));compact_device=nullptr;}
  entries_.emplace(std::make_pair(layer,expert),entry);used_+=bytes;arena.used+=bytes;
  if(compact_uploaded){output16_counts_v252_[0]++;output16_counts_v252_[1]+=compact_uploaded;output16_counts_v252_[2]+=inverse_bytes;output16_counts_v252_[3]=std::max(output16_counts_v252_[3],uint64_t(max_compact));}
 }catch(...){if(compact_device){cudaStreamSynchronize(0);cudaFree(compact_device);}cudaFree(entry.storage);throw;}
}
NativeGpuExperts::NativeGpuExperts(std::shared_ptr<const BorrowedWeights>w,size_t requested,const std::vector<std::array<uint64_t,384>>&counts):weights_(std::move(w)){
 if(!weights_||counts.size()!=40||requested<(size_t(1)<<28)||requested>(size_t(22)<<30))throw std::invalid_argument("invalid native GPU expert budget");
 std::vector<int>devices;
 const char*multiple=std::getenv("DS41_NATIVE_EXPERT_DEVICES");
 const char*single=std::getenv("DS41_NATIVE_EXPERT_DEVICE");
 if(multiple&&single)throw std::invalid_argument("choose one explicit expert device setting");
 std::string selection=multiple?multiple:single?single:"0";
 if(selection.empty()||selection.back()==',')throw std::invalid_argument("empty expert device selection");
 std::istringstream stream(selection);std::string item;std::set<int>seen;
 while(std::getline(stream,item,',')){
  if(item.size()!=1||item[0]<'0'||item[0]>'3'||!seen.insert(item[0]-'0').second)throw std::invalid_argument("expert devices must be unique visible indices0..3");
  devices.push_back(item[0]-'0');
 }
 int visible=0;check(cudaGetDeviceCount(&visible));workspace_bytes_=256*1024;
 const size_t reserve=(size_t(1)<<30)+workspace_bytes_+1024*1024;
 try{
  for(int device:devices){
   if(device>=visible)throw std::invalid_argument("expert device is not visible");
   ExpertDeviceScope scope(device);size_t free,total;check(cudaMemGetInfo(&free,&total));
   if(free<=reserve)throw std::runtime_error("no free native expert cache budget");
   arenas_.push_back(Arena{});auto&a=arenas_.back();a.device=device;a.budget=std::min(requested,free-reserve);budget_+=a.budget;
   check(cudaMalloc(&a.x,8*5120*2));check(cudaMalloc(&a.alpha,8*6*4));check(cudaMalloc(&a.output,8*6*5120*2));check(cudaMalloc(&a.workspace,workspace_bytes_));check(cudaStreamCreateWithFlags(&a.stream,cudaStreamNonBlocking));check(cudaHostAlloc(&a.host_output,8*6*5120*2,cudaHostAllocDefault));
  }
  struct Candidate{int layer,expert;uint64_t count;size_t bytes;};
  std::vector<Candidate>ranked;
  for(int layer=0;layer<40;layer++)for(int expert=0;expert<384;expert++)if(counts[layer][expert]){
   size_t bytes=0;
   for(const char*which:{"w1","w3","w2"}){
    const bool down=std::strcmp(which,"w2")==0;
    const auto p=weights_->inspect_projection("layers."+std::to_string(layer)+".ffn.experts."+std::to_string(expert)+"."+which,down?5120:2304,down?2304:5120);
    bytes=(bytes+255)&~size_t(255);bytes+=size_t(p.rows)*p.weight_row_bytes;
    if(p.scale_row_bytes){bytes=(bytes+255)&~size_t(255);bytes+=size_t(p.rows)*p.scale_row_bytes;}
   }
   if(!bytes)throw std::runtime_error("empty expert cache candidate");
   ranked.push_back({layer,expert,counts[layer][expert],bytes});
  }
  // Spend VRAM on the most prompt uses per stored byte, across all layers.
  std::stable_sort(ranked.begin(),ranked.end(),[](const Candidate&a,const Candidate&b){
   return static_cast<long double>(a.count)*b.bytes>static_cast<long double>(b.count)*a.bytes;
  });
  size_t next=0;
  for(const auto&candidate:ranked){
   bool fits=false;for(const auto&arena:arenas_)if(candidate.bytes<=arena.budget-arena.used){fits=true;break;}
   if(fits)load(candidate.layer,candidate.expert,int(next++%arenas_.size()));
  }
  // Warm each selected card before the target stream waits on the CPU doorbell.
  for(size_t i=0;i<arenas_.size();++i){
   auto&a=arenas_[i];ExpertDeviceScope scope(a.device);
   auto found=std::find_if(entries_.begin(),entries_.end(),[&](const auto&e){return e.second.arena==int(i);});
   if(found==entries_.end())throw std::runtime_error("selected expert card has no cached weights");
   check(cudaMemset(a.x,0,5120*2));float alpha=1.f;check(cudaMemcpy(a.alpha,&alpha,4,cudaMemcpyHostToDevice));
   const auto&p=found->second.projections;
   native_expert_cached_gemv(a.x,a.alpha,10.f,p[0],p[1],p[2],a.output,a.workspace,workspace_bytes_,a.stream);check(cudaStreamSynchronize(a.stream));
  }
 }catch(...){release();throw;}
}
void NativeGpuExperts::release()noexcept{
 int previous=0;cudaGetDevice(&previous);
 for(size_t i=0;i<arenas_.size();++i){auto&a=arenas_[i];cudaSetDevice(a.device);if(a.stream)cudaStreamSynchronize(a.stream);
  for(auto&v:entries_)if(v.second.arena==int(i))cudaFree(v.second.storage);
  cudaFree(a.x);cudaFree(a.alpha);cudaFree(a.output);cudaFree(a.workspace);if(a.host_output)cudaFreeHost(a.host_output);if(a.stream)cudaStreamDestroy(a.stream);
 }
 entries_.clear();arenas_.clear();cudaSetDevice(previous);
}
NativeGpuExperts::~NativeGpuExperts(){release();}
std::vector<std::array<uint64_t,4>>NativeGpuExperts::device_statistics()const{
 std::vector<std::array<uint64_t,4>>result;
 for(const auto&a:arenas_)result.push_back({uint64_t(a.device),a.hits,a.used,a.budget});
 return result;
}
std::vector<uint16_t>NativeGpuExperts::run(NativeExperts&cpu,int layer,const std::vector<uint16_t>&input,const std::vector<int64_t>&ids,const std::vector<float>&alpha){
 if(input.size()!=5120||ids.size()!=6||alpha.size()!=6)throw std::invalid_argument("native hybrid needs exactly one six-expert token");
 std::array<int,6>rank{0,1,2,3,4,5};std::sort(rank.begin(),rank.end(),[&](int a,int b){return ids[a]<ids[b];});
 std::vector<int64_t>miss_ids;std::vector<float>miss_alpha;std::vector<int>miss_slots;
 struct Batch {std::array<NativeDeviceExpert,6>experts{};std::array<float,6>alpha{};std::vector<int>slots;};
 std::vector<Batch>batches(arenas_.size());
 for(int slot=0;slot<6;slot++){
  int index=rank[slot];if(ids[index]<0||ids[index]>=384||!std::isfinite(alpha[index])||alpha[index]<0||(slot&&ids[index]==ids[rank[slot-1]]))throw std::invalid_argument("invalid native hybrid routes");
  auto found=entries_.find({layer,int(ids[index])});
  if(found==entries_.end()){miss_ids.push_back(ids[index]);miss_alpha.push_back(alpha[index]);miss_slots.push_back(slot);misses_++;}
  else{int at=found->second.arena;auto&batch=batches[at];size_t i=batch.slots.size();
   std::copy_n(found->second.projections,3,batch.experts[i].projection);batch.alpha[i]=alpha[index];batch.slots.push_back(slot);hits_++;arenas_[at].hits++;}
 }
 for(size_t at=0;at<arenas_.size();++at){auto&batch=batches[at];if(batch.slots.empty())continue;auto&a=arenas_[at];ExpertDeviceScope scope(a.device);
  check(cudaMemcpyAsync(a.x,input.data(),5120*2,cudaMemcpyHostToDevice,a.stream));
  if(batch.slots.size()>1)native_expert_cached_many(a.x,batch.experts.data(),batch.alpha.data(),int(batch.slots.size()),10.f,a.output,a.workspace,workspace_bytes_,a.stream);
  else{check(cudaMemcpyAsync(a.alpha,batch.alpha.data(),4,cudaMemcpyHostToDevice,a.stream));const auto&p=batch.experts[0].projection;native_expert_cached_gemv(a.x,a.alpha,10.f,p[0],p[1],p[2],a.output,a.workspace,workspace_bytes_,a.stream);}
 }
 std::vector<uint16_t>result(6*5120);
 if(!miss_ids.empty()){
  auto out=cpu.run_selected(layer,1,input,miss_ids,miss_alpha,int(miss_ids.size()));
  for(size_t i=0;i<miss_slots.size();i++)std::copy_n(out.data()+i*5120,5120,result.data()+miss_slots[i]*5120);
 }
 for(size_t at=0;at<arenas_.size();++at){const auto&batch=batches[at];if(batch.slots.empty())continue;auto&a=arenas_[at];ExpertDeviceScope scope(a.device);
  std::vector<uint16_t>outputs(batch.slots.size()*5120);check(cudaMemcpyAsync(a.host_output,a.output,outputs.size()*2,cudaMemcpyDeviceToHost,a.stream));check(cudaStreamSynchronize(a.stream));std::copy_n(a.host_output,outputs.size(),outputs.data());
  for(size_t i=0;i<batch.slots.size();++i)std::copy_n(outputs.data()+i*5120,5120,result.data()+batch.slots[i]*5120);
 }
 return result;
}
// GPU hits run while one masked CPU batch retains weight reuse across all rows.
// Zero routing coefficients skip GPU-covered slots in the v54 CPU task library.
std::vector<uint16_t>NativeGpuExperts::run_many(NativeExperts&cpu,int layer,const std::vector<uint16_t>&input,const std::vector<int64_t>&ids,const std::vector<float>&alpha){
 const int rows=int(input.size()/5120);
 static std::atomic<int>traces{0};const bool debug=std::getenv("DS41_HYBRID_TRACE")&&traces.fetch_add(1)<48;
 auto trace=[&](const char*stage,int item=-1){if(debug){std::fprintf(stderr,"HYBRID52 layer=%d rows=%d stage=%s item=%d\n",layer,rows,stage,item);std::fflush(stderr);}};trace("enter");
 if(rows<1||rows>8||input.size()!=size_t(rows*5120)||ids.size()!=size_t(rows*6)||alpha.size()!=ids.size())throw std::invalid_argument("bounded hybrid multirow geometry required");
 for(auto value:input)if((value&0x7f80)==0x7f80)throw std::invalid_argument("nonfinite hybrid input");
 struct Batch{std::array<NativeDeviceExpert,6>experts{};std::array<float,6>alpha{};std::vector<int>slots;};
 std::vector<std::array<Batch,8>>batches(arenas_.size());
 std::vector<float>cpu_alpha=alpha;bool has_cpu_misses=false;
 for(int row=0;row<rows;row++){
  std::array<int,6>rank{0,1,2,3,4,5};
  std::sort(rank.begin(),rank.end(),[&](int a,int b){return ids[row*6+a]<ids[row*6+b];});
  for(int slot=0;slot<6;slot++){
   const int index=row*6+rank[slot];
   if(ids[index]<0||ids[index]>=384||!std::isfinite(alpha[index])||alpha[index]<0||(slot&&ids[index]==ids[row*6+rank[slot-1]]))throw std::invalid_argument("invalid hybrid multirow routes");
   auto found=entries_.find({layer,int(ids[index])});
   if(found==entries_.end()){has_cpu_misses=true;misses_++;}
   else{cpu_alpha[index]=0.f;const int at=found->second.arena;auto&batch=batches[at][row];const size_t position=batch.slots.size();
    std::copy_n(found->second.projections,3,batch.experts[position].projection);batch.alpha[position]=alpha[index];batch.slots.push_back(slot);hits_++;arenas_[at].hits++;}
  }
 }
 for(size_t at=0;at<arenas_.size();at++){
  auto&a=arenas_[at];trace("device_scope",int(at));ExpertDeviceScope scope(a.device);
  for(int row=0;row<rows;row++){
   auto&batch=batches[at][row];if(batch.slots.empty())continue;
   auto*x=a.x+row*5120;auto*out=a.output+row*6*5120;
   trace("GPU_H2D_before",row);check(cudaMemcpyAsync(x,input.data()+row*5120,5120*2,cudaMemcpyHostToDevice,a.stream));trace("GPU_H2D_after",row);
   if(batch.slots.size()>1)native_expert_cached_many(x,batch.experts.data(),batch.alpha.data(),int(batch.slots.size()),10.f,out,a.workspace,workspace_bytes_,a.stream);
   else{auto*coefficient=a.alpha+row*6;check(cudaMemcpyAsync(coefficient,batch.alpha.data(),4,cudaMemcpyHostToDevice,a.stream));const auto&p=batch.experts[0].projection;native_expert_cached_gemv(x,coefficient,10.f,p[0],p[1],p[2],out,a.workspace,workspace_bytes_,a.stream);}
  }
 }
 std::vector<uint16_t>result(rows*6*5120);
 trace("GPU_enqueued");
 if(has_cpu_misses){
  trace("CPU_mask_batch_before");result=cpu.run(layer,rows,input,ids,cpu_alpha);trace("CPU_mask_batch_after");
 }
 for(size_t at=0;at<arenas_.size();at++){
  auto&a=arenas_[at];trace("device_scope",int(at));ExpertDeviceScope scope(a.device);
  for(int row=0;row<rows;row++){
   const auto&batch=batches[at][row];if(batch.slots.empty())continue;std::vector<uint16_t>outputs(batch.slots.size()*5120);
   trace("GPU_D2H_before",row);check(cudaMemcpyAsync(a.host_output+row*6*5120,a.output+row*6*5120,outputs.size()*2,cudaMemcpyDeviceToHost,a.stream));check(cudaStreamSynchronize(a.stream));std::copy_n(a.host_output+row*6*5120,outputs.size(),outputs.data());trace("GPU_D2H_after",row);
   for(size_t j=0;j<batch.slots.size();j++)std::copy_n(outputs.data()+j*5120,5120,result.data()+(row*6+batch.slots[j])*5120);
  }
 }
 trace("leave");return result;
}

}
