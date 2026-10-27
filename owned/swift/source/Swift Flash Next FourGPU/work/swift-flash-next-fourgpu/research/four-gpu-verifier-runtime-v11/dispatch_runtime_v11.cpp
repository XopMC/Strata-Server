#include "../four-gpu-expert-dispatch-v10-r2/dispatch_kernels_v10.cuh"
#include "strata/kernels/iq_kernels.hpp"
#include "peer_matrix.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>
namespace four_dispatch_v10 { namespace {
namespace N=strata::kernels;
void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
void ck(cudaError_t e,const char*s){if(e!=cudaSuccess)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(e));}
[[noreturn]]void fatal(){std::fprintf(stderr,"V10 fatal unchecked dispatcher retirement\n");std::_Exit(2);}
struct Current {int device=-1;Current(){ck(cudaGetDevice(&device),"save context");}~Current(){if(cudaSetDevice(device)!=cudaSuccess)fatal();}};
struct Body {
 int device;cudaGraph_t graph=nullptr;
 Body(int d,cudaStream_t stream,const std::function<void()>&fn):device(d){ck(cudaSetDevice(d),"body context");bool capture=false;try{ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"body capture");capture=true;fn();ck(cudaStreamEndCapture(stream,&graph),"body capture end");capture=false;}catch(...){if(capture){cudaGraph_t discard=nullptr;cudaStreamEndCapture(stream,&discard);if(discard)cudaGraphDestroy(discard);}throw;}
 }
 ~Body(){if(graph&&(cudaSetDevice(device)!=cudaSuccess||cudaGraphDestroy(graph)!=cudaSuccess))fatal();}
};
void append(cudaStream_t stream,const std::vector<cudaGraph_t>&bodies){
 cudaStreamCaptureStatus status;cudaGraph_t parent=nullptr;const cudaGraphNode_t*borrowed=nullptr;const cudaGraphEdgeData*edge_data=nullptr;size_t n=0;
 ck(cudaStreamGetCaptureInfo(stream,&status,nullptr,&parent,&borrowed,&edge_data,&n),"complete pre frontier");need(status==cudaStreamCaptureStatusActive&&parent&&!bodies.empty(),"V10 append requires owner capture");
 std::vector<cudaGraphNode_t>deps,ends;std::vector<cudaGraphEdgeData>edges;if(n){deps.assign(borrowed,borrowed+n);if(edge_data)edges.assign(edge_data,edge_data+n);else edges.resize(n);}
 // Each child receives exactly the same full frontier. CUDA retains each
 // source kernel's context; cloned children live in one ordinary host DAG.
 for(auto body:bodies){cudaGraphNodeParams args{};args.type=cudaGraphNodeTypeGraph;args.graph.graph=body;args.graph.ownership=cudaGraphChildGraphOwnershipClone;cudaGraphNode_t child=nullptr;ck(cudaGraphAddNode(&child,parent,n?deps.data():nullptr,n?edges.data():nullptr,n,&args),"four-bank numerical child");ends.push_back(child);}
 ck(cudaStreamUpdateCaptureDependencies(stream,ends.data(),nullptr,ends.size(),cudaStreamSetCaptureDependencies),"full four-bank join frontier");
}
void pointer(const void*p,int d,size_t alignment,const char*what){need(p&&!(uintptr_t(p)&(alignment-1)),what);cudaPointerAttributes a{};ck(cudaPointerGetAttributes(&a,p),what);need(a.type==cudaMemoryTypeDevice&&a.device==d,"V10 borrowed pointer ownership mismatch");}
}
struct Dispatcher::State {
 OwnerView owner;std::array<BankBuffers,4>banks;std::shared_ptr<const void>lease;CpuPolicy policy;
 std::unique_ptr<Body>pre,native_gather,cpu_gather;std::array<std::unique_ptr<Body>,4>bank_body;
 uint64_t expected=0,last=0;bool active=false,closed=false,poison=false;int phase=0;
 State(OwnerView o,const std::array<BankBuffers,4>&b,std::shared_ptr<const void>l,CpuPolicy p):owner(o),banks(b),lease(std::move(l)),policy(p){
  Current restore;need(lease&&o.device>=0&&o.device<2&&o.layer/24==o.device&&o.layer>=0&&o.layer<48&&o.T>=1&&o.T<=8&&o.stream,"V10 requires two original dense owners and one valid layer/T");std::string error;const bool peers=strata_peer_default::require_all(4,error);need(peers,error.c_str());
  const char*act=std::getenv("STRATA_SWIFT_NVFP4_ACTSTAGE_V264");need(act&&!std::strcmp(act,"0")&&!N::iq_old_kernels(),"V10 requires selected native40 ACTSTAGE0 leaves");
  for(const char*name:{"STRATA_GU_SWIGLU","STRATA_GROUPED_V1","STRATA_OLD_IQ_MMVQ","STRATA_EXP_MODE","STRATA_EXPDEC"}){const char*v=std::getenv(name);need(!v||!std::strcmp(v,"0"),"V10 unsupported native arithmetic policy");}
  ck(cudaSetDevice(o.device),"owner pointers");pointer(o.inputs,o.device,8,"input binding");pointer(o.original_x,o.device,16,"original FP32 input");pointer(o.original_native_q8,o.device,16,"original native Q8 input");pointer(o.parts,o.device,16,"original parts");pointer(o.native_hits,o.device,16,"original native hits");pointer(o.mask,o.device,8,"native mask");pointer(o.receipt,o.device,8,"layer receipt");pointer(o.stage_error,o.device,4,"whole stage error");
  for(int d=0;d<4;++d){const auto&x=b[d];need(x.capture_stream&&x.native_scratch_bytes>=N::native_expert_scratch_bytes(80,F),"V10 native scratch capacity");if(policy==CpuPolicy::Compact)need(x.compact&&x.persistent_ctas>0&&x.persistent_ctas<=2048,"V10 explicit compact grid");ck(cudaSetDevice(d),"bank pointers");pointer(x.cpu.plan,d,8,"bank plan");pointer(x.cpu.plan_error,d,4,"bank plan error");pointer(x.cpu.physical,d,8,"physical view");pointer(x.cpu.input,d,16,"local FP32 input");pointer(x.cpu.output,d,16,"local CPU rows");pointer(x.cpu.status,d,8,"CPU status");pointer(x.native_q8,d,16,"local native Q8");pointer(x.native_plan,d,16,"native grouped plan");pointer(x.native_scratch,d,16,"native scratch");pointer(x.native_rows,d,16,"native raw rows");pointer(x.status,d,8,"bank completion");}
  ck(cudaSetDevice(o.device),"initialize receipt");ck(cudaMemsetAsync(o.receipt,0,sizeof(LayerReceipt),o.stream),"receipt clear");ck(cudaStreamSynchronize(o.stream),"positive receipt initialization");
  const auto L=N::native_expert_layout(40,40,H,F);need(L.bytes==four_banks_v3::BlobBytes&&!L.had_block,"V10 unchanged native layout");
  pre=std::make_unique<Body>(o.device,o.stream,[&]{ck(cpu_rows_v5::plan_async(o.inputs,const_cast<cpu_rows_v5::Plan*>(b[o.device].cpu.plan),const_cast<uint32_t*>(b[o.device].cpu.plan_error),o.stream),"V10 original logical plan");ck(detail::resolve_native(const_cast<cpu_rows_v5::Plan*>(b[o.device].cpu.plan),o.inputs,const_cast<uint32_t*>(b[o.device].cpu.plan_error),o.mask,o.stage_error,o.layer,o.T,o.stream),"V10 physical resolution for native subset");});
  for(int d=0;d<4;++d){auto x=b[d];bank_body[d]=std::make_unique<Body>(d,x.capture_stream,[&,d,x]{ck(detail::copy_inputs(o,x,b[o.device],x.capture_stream),"V10 exact original inputs/plan transport");ck(detail::native_plan(x,d,x.capture_stream),"V10 original-order native groups");auto*np=x.native_plan;
    N::native_expert_grouped(L,np->ptr,np->start,&np->groups,np->original,np->tok,o.T*10,o.T*10,x.native_q8,x.native_scratch,x.native_rows,x.capture_stream);
    if(policy==CpuPolicy::Compact)ck(cpu_compact_v8::chain(x.cpu,x.compact,d,x.persistent_ctas,x.capture_stream),"V10 explicit CPU compact semantics");else ck(cpu_grouped_v6::chain(x.cpu,d,x.capture_stream),"V10 CPU baseline semantics");
    ck(detail::bank_commit(x,d,x.capture_stream),"V10 complete native plus CPU bank receipt");});}
  native_gather=std::make_unique<Body>(o.device,o.stream,[&]{ck(detail::gather_native(o,b,o.stream),"V10 raw native row gather");});
  cpu_gather=std::make_unique<Body>(o.device,o.stream,[&]{ck(detail::gather_cpu(o,b,o.stream),"V10 CPU row gather after mapped copy");});
 }
 Rows rows()const{return {owner.mask->rows,&owner.mask->count,&owner.mask->error};}
 void capture_owner(){int d=-1;ck(cudaGetDevice(&d),"capture current context");need(d==owner.device&&!closed&&!poison&&!active,"V10 capture scope/lifecycle");}
};
Dispatcher::Dispatcher(OwnerView o,const std::array<BankBuffers,4>&b,std::shared_ptr<const void>l,CpuPolicy p):p_(std::make_shared<State>(o,b,std::move(l),p)){}
Dispatcher::~Dispatcher(){if(p_){try{close();}catch(...){fatal();}}}
std::shared_ptr<const void>Dispatcher::retain()const{need(p_&&!p_->closed,"V10 retain closed source");return p_;}
Rows Dispatcher::capture_pre(){auto&p=*p_;p.capture_owner();need(p.phase==0,"V10 pre order");append(p.owner.stream,{p.pre->graph});p.phase=1;return p.rows();}
Rows Dispatcher::capture_post(){auto&p=*p_;p.capture_owner();need(p.phase==1,"V10 post order");std::vector<cudaGraph_t>g;for(auto&b:p.bank_body)g.push_back(b->graph);append(p.owner.stream,g);append(p.owner.stream,{p.native_gather->graph});p.phase=2;return p.rows();}
void Dispatcher::capture_cpu_rows(){auto&p=*p_;p.capture_owner();need(p.phase==2,"V10 CPU rows must follow post and original mapped-copy frontier");append(p.owner.stream,{p.cpu_gather->graph});p.phase=0;}
void Dispatcher::begin(uint64_t sequence){auto&p=*p_;need(!p.closed&&!p.poison&&!p.active&&p.phase==0&&sequence>p.last,"V10 requires new monotonic drained frame");p.expected=p.last=sequence;p.active=true;}
bool Dispatcher::finish(std::string&error){auto&p=*p_;need(p.active&&!p.closed,"V10 finish without begin");Current restore;ck(cudaSetDevice(p.owner.device),"finish owner");ck(cudaStreamSynchronize(p.owner.stream),"positive complete multi-context parent DAG");LayerReceipt receipt{};ck(cudaMemcpyAsync(&receipt,p.owner.receipt,sizeof receipt,cudaMemcpyDeviceToHost,p.owner.stream),"layer receipt readback");ck(cudaStreamSynchronize(p.owner.stream),"positive layer receipt readback");p.active=false;bool ok=receipt.done==0xa10u&&receipt.sequence==p.expected&&receipt.T==uint32_t(p.owner.T)&&receipt.layer==uint32_t(p.owner.layer)&&!receipt.error;if(!ok){p.poison=true;error="V10 combined bank layer receipt refused";}return ok;}
bool Dispatcher::finish_all(const std::vector<Dispatcher*>&all,std::string&error){
 need(!all.empty()&&all.size()<=24,"V10r1 bounded finish group");need(all.front()&&all.front()->p_,"V11 missing first finish member");auto&first=*all.front()->p_;need(first.active&&!first.closed,"V10r1 finish without begin");
 for(size_t i=0;i<all.size();++i){need(all[i]&&all[i]->p_,"V10r1 missing finish member");auto&p=*all[i]->p_;need(p.active&&!p.closed&&p.owner.device==first.owner.device&&p.owner.stream==first.owner.stream&&p.expected==first.expected&&p.owner.T==first.owner.T&&p.owner.layer==first.owner.layer+int(i)&&p.owner.receipt==first.owner.receipt+i,"V10r1 finish requires contiguous same-stage current receipts");}
 Current restore;ck(cudaSetDevice(first.owner.device),"finish stage owner");ck(cudaStreamSynchronize(first.owner.stream),"positive complete stage host DAG");std::vector<LayerReceipt>receipts(all.size());ck(cudaMemcpyAsync(receipts.data(),first.owner.receipt,receipts.size()*sizeof(LayerReceipt),cudaMemcpyDeviceToHost,first.owner.stream),"one bounded stage receipt readback");ck(cudaStreamSynchronize(first.owner.stream),"positive stage receipt readback");bool clean=true;
 for(size_t i=0;i<all.size();++i){auto&p=*all[i]->p_;const auto&r=receipts[i];const bool ok=r.done==0xa10u&&r.sequence==p.expected&&r.T==uint32_t(p.owner.T)&&r.layer==uint32_t(p.owner.layer)&&!r.error;p.active=false;if(!ok)clean=false;}
 if(!clean){for(auto*d:all)d->p_->poison=true;error="V10r1 complete stage layer receipts refused";}return clean;
}
void Dispatcher::abort(){auto&p=*p_;Current restore;p.poison=true;ck(cudaSetDevice(p.owner.device),"abort owner");ck(cudaStreamSynchronize(p.owner.stream),"abort positive graph drain");p.active=false;}
void Dispatcher::close(){if(!p_)return;need(p_.use_count()==1,"V10 parent graph retain tokens remain live");auto&p=*p_;need(!p.active&&p.phase==0,"V10 close before checked drain/capture completion");Current restore;ck(cudaSetDevice(p.owner.device),"close owner");ck(cudaStreamSynchronize(p.owner.stream),"close positive original stream drain");p.closed=true;p_.reset();}
}
