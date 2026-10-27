#include "strata_owned/device_admission.hpp"
// HOST ONLY: unchanged dequant_f16 and one screened cuBLASLt algorithm.
#include "../prefill-output-lt-v301/output_lt_v301.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>

static_assert(CUBLAS_STATUS_INVALID_VALUE==7, "R1 actual reserved-capability status must remain7");
namespace swift_output_lt_v301 {
namespace {
constexpr int T=32768,N=2560,K=6144;
constexpr int64_t ScratchElems=32ll<<20;
constexpr size_t WorkspaceBytes=32u<<20;
void need(bool x,const char* e){if(!x)throw std::runtime_error(e);}
void ck(cudaError_t e,const char* where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
void cb(cublasStatus_t e,const char* where){if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string(where)+": Lt status "+std::to_string(int(e)));}
template<class U>U config(const cublasLtMatmulAlgo_t&a,cublasLtMatmulAlgoConfigAttributes_t which){U v{};size_t bytes=0;cb(cublasLtMatmulAlgoConfigGetAttribute(&a,which,&v,sizeof v,&bytes),"Lt config");need(bytes==sizeof v,"Lt config size");return v;}
template<class U>U cap(const cublasLtMatmulAlgo_t&a,cublasLtMatmulAlgoCapAttributes_t which){U v{};size_t bytes=0;cb(cublasLtMatmulAlgoCapGetAttribute(&a,which,&v,sizeof v,&bytes),"Lt capability");need(bytes==sizeof v,"Lt cap size");return v;}
bool aligned(const void*p,size_t n){return p&&n&&reinterpret_cast<uintptr_t>(p)%n==0;}
// Same-version documented serialization: restore all eight words verbatim.
// No mask, normalization, heuristic reselection or meaning assigned to opaque bits.
std::string restore_policy_r2(cublasLtMatmulAlgo_t&algo,int device){
 static_assert(sizeof(algo.data)==8*sizeof(uint64_t));
 const char*dir=std::getenv("STRATA_PP_OUTPUT_LT_V301_POLICY_DIR");need(dir&&*dir,"V301 model requires explicit component-qualified policy directory");
 std::ifstream in(std::string(dir)+"/cuda"+std::to_string(device)+".policy",std::ios::binary);need(bool(in),"V301 component policy absent");
 char text[512]{};in.getline(text,sizeof text);need(!in.fail()&&in.eof()==false&&in.peek()==EOF,"V301 R2 bounded single newline-terminated policy required");
 std::string line(text);std::ostringstream prefix;prefix<<"V301_POLICY_1 CUDA"<<device<<" T32768 N2560 K6144 runtime="<<CUDART_VERSION<<" Lt="<<cublasLtGetVersion()<<" algo=";
 const std::string p=prefix.str(),tail=" custom=0 swizzle=1 align=16,16,16,16,";
 need(line.size()==p.size()+128+tail.size()&&line.compare(0,p.size(),p)==0&&line.compare(p.size()+128,tail.size(),tail)==0,"V301 R2 policy version/device/layout/config/alignment mismatch");
 for(size_t word=0;word<8;word++){uint64_t value=0;for(size_t i=0;i<16;i++){const char c=line[p.size()+word*16+i];need((c>='0'&&c<='9')||(c>='a'&&c<='f'),"V301 R2 canonical lowercase 64-bit hex required");value=(value<<4)|uint64_t(c<='9'?c-'0':c-'a'+10);}algo.data[word]=value;}
 return line+'\n';
}
void validate_public_r2(const cublasLtMatmulAlgo_t&algo){
 // All nine installed public configuration attributes, including uint16 shape IDs.
 const uint32_t expected[9]={5,20,1,0,1,0,7,0,0};
 for(int id=0;id<9;id++){const auto attr=static_cast<cublasLtMatmulAlgoConfigAttributes_t>(id);if(id<7)need(config<uint32_t>(algo,attr)==expected[id],"V301 R2 public config differs");else need(config<uint16_t>(algo,attr)==expected[id],"V301 R2 public shape config differs");}
 // Bounded capability identity from actual V309 metadata, independently screened
 // by the mandatory restored-object component before any model acceptance.
 const char* expected_caps[24]={"01000000","07000000","01000000","01000000","01000000","00000000","14000000170000001800000015000000","00000000",nullptr,nullptr,"00000000","03000000","ff030000","0700000000000000","00000000","0202010000000000","10000000","10000000","10000000","10000000",nullptr,"01000000","00000000","00000000"};
 for(int id=0;id<24;id++){
  const auto attr=static_cast<cublasLtMatmulAlgoCapAttributes_t>(id);size_t bytes=0;
  const auto status=cublasLtMatmulAlgoCapGetAttribute(&algo,attr,nullptr,0,&bytes);
  if(!expected_caps[id]){need(status==CUBLAS_STATUS_INVALID_VALUE&&bytes==0,"V301 R2 unsupported capability changed");continue;}
  const size_t n=std::strlen(expected_caps[id])/2;need(status==CUBLAS_STATUS_SUCCESS&&bytes==n&&n<=16,"V301 R2 capability extent/status differs");
  unsigned char value[16]{};size_t written=0;cb(cublasLtMatmulAlgoCapGetAttribute(&algo,attr,value,n,&written),"R2 restored capability");need(written==n,"V301 R2 capability size differs");
  const char*hex="0123456789abcdef";for(size_t i=0;i<n;i++)need(hex[value[i]>>4]==expected_caps[id][2*i]&&hex[value[i]&15]==expected_caps[id][2*i+1],"V301 R2 capability bytes differ");
 }
}
void abort_capture_and_drain(cudaStream_t stream)noexcept{if(!stream)return;cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;if(cudaStreamIsCapturing(stream,&status)==cudaSuccess&&status!=cudaStreamCaptureStatusNone){cudaGraph_t graph=nullptr;cudaStreamEndCapture(stream,&graph);if(graph)cudaGraphDestroy(graph);}cudaStreamSynchronize(stream);}
}
bool requested(){static const bool on=[](){const char*v=std::getenv("STRATA_PP_OUTPUT_LT_V301");if(!v||!std::strcmp(v,"0"))return false;need(!std::strcmp(v,"1"),"V301 flag must be absent/0 or exact1");return true;}();return on;}
struct Context::State {
 int device=-1;cudaStream_t stream=nullptr;uint16_t* scratch=nullptr;int64_t elems=0;void* workspace=nullptr;size_t bytes=0;
 cublasLtHandle_t handle=nullptr;cublasLtMatmulDesc_t op=nullptr;cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;cublasLtMatmulPreference_t preference=nullptr;
 cublasLtMatmulAlgo_t algo{};bool ready=false,component_qualification=false;uint64_t calls[2]{},fallbacks[2]{},errors=0;std::array<uint32_t,4> alignment{};
 std::string policy()const{std::ostringstream o;o<<"V301_POLICY_1 CUDA"<<device<<" T32768 N2560 K6144 runtime="<<CUDART_VERSION<<" Lt="<<cublasLtGetVersion()<<" algo="<<std::hex<<std::setfill('0');for(auto word:algo.data)o<<std::setw(16)<<word;o<<std::dec<<" custom="<<config<int>(algo,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)<<" swizzle="<<config<unsigned>(algo,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)<<" align=";for(auto n:alignment)o<<n<<',';o<<'\n';return o.str();}
 ~State(){
  const strata::core::OnDevice on(device);abort_capture_and_drain(stream);
  if(ready)std::fprintf(stderr,"V301 SUMMARY CUDA%d GDN=%llu QSA=%llu fallbackGDN=%llu fallbackQSA=%llu errors=%llu\n",device,(unsigned long long)calls[0],(unsigned long long)calls[1],(unsigned long long)fallbacks[0],(unsigned long long)fallbacks[1],(unsigned long long)errors);
  if(preference)cublasLtMatmulPreferenceDestroy(preference);if(c)cublasLtMatrixLayoutDestroy(c);if(b)cublasLtMatrixLayoutDestroy(b);if(a)cublasLtMatrixLayoutDestroy(a);if(op)cublasLtMatmulDescDestroy(op);if(handle)cublasLtDestroy(handle);
  // Borrowed scratch/workspace are never freed by this context.
 }
};
Context::Context()=default;Context::~Context()=default;
bool Context::initialize(int device,void* stream,uint16_t* scratch,int64_t elems,void* workspace,size_t bytes,std::string& err,bool component_qualification){
 try{
  need(!s_,"V301 already initialized");need(requested(),"V301 initialization while flagOFF");
  int actual=-1;ck(cudaGetDevice(&actual),"owning device");need(actual==device&&(device==0||device==1),"V301 requires owning selected CUDA0/1 context");
  cudaStreamCaptureStatus capture;ck(cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture),"init capture check");need(capture==cudaStreamCaptureStatusNone,"V301 initialization inside capture forbidden");
  cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,device),"SM89 properties");need(prop.major==8&&prop.minor==9,"V301 SM89 only");
  const auto*u=reinterpret_cast<const unsigned char*>(prop.uuid.bytes);char uuid[64];std::snprintf(uuid,sizeof uuid,"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
  std::string admission_error;const bool admitted=strata_owned::require_selected_pair(admission_error);need(admitted,admission_error.c_str());
  int runtime=0;ck(cudaRuntimeGetVersion(&runtime),"runtime version");need(runtime==CUDART_VERSION&&cublasLtGetVersion()==CUBLAS_VERSION,"V301 runtime/header library version mismatch");
  auto candidate=std::make_unique<State>();candidate->device=device;candidate->stream=static_cast<cudaStream_t>(stream);candidate->component_qualification=component_qualification;need(candidate->stream,"V301 explicit stream required");
  cb(cublasLtCreate(&candidate->handle),"Lt create");cb(cublasLtMatmulDescCreate(&candidate->op,CUBLAS_COMPUTE_32F,CUDA_R_32F),"FP32 compute/scale");
  cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;cb(cublasLtMatmulDescSetAttribute(candidate->op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof ta),"transposeA");cb(cublasLtMatmulDescSetAttribute(candidate->op,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof tb),"transposeB");
  cb(cublasLtMatrixLayoutCreate(&candidate->a,CUDA_R_16F,K,N,K),"FP16 weight layout");cb(cublasLtMatrixLayoutCreate(&candidate->b,CUDA_R_16F,K,T,K),"FP16 input layout");cb(cublasLtMatrixLayoutCreate(&candidate->c,CUDA_R_32F,N,T,N),"FP32 output layout");
  cb(cublasLtMatmulPreferenceCreate(&candidate->preference),"preference");const size_t ws=WorkspaceBytes;cb(cublasLtMatmulPreferenceSetAttribute(candidate->preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof ws),"existing32MiB workspace limit");
  std::string restored_policy;
  if(component_qualification){
   // Same bounded8 results as V154. No timing/search, one startup selection;
  // no heuristic/API configuration is performed in run()/capture/relayout.
  std::array<cublasLtMatmulHeuristicResult_t,8> results{};int got=0,selected=0;
  cb(cublasLtMatmulAlgoGetHeuristic(candidate->handle,candidate->op,candidate->a,candidate->b,candidate->c,candidate->c,candidate->preference,results.size(),results.data(),&got),"startup8 heuristics");
  for(int i=0;i<got;i++){
   const auto&r=results[i];if(r.state!=CUBLAS_STATUS_SUCCESS||r.workspaceSize!=0)continue;
   const auto&a=r.algo;
   if(config<int>(a,CUBLASLT_ALGO_CONFIG_ID)!=5||config<unsigned>(a,CUBLASLT_ALGO_CONFIG_TILE_ID)!=20||config<unsigned>(a,CUBLASLT_ALGO_CONFIG_STAGES_ID)!=7||config<int>(a,CUBLASLT_ALGO_CONFIG_SPLITK_NUM)!=1||config<unsigned>(a,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)!=0||cap<uint64_t>(a,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS)!=0x10202)continue;
   cublasLtMatmulHeuristicResult_t checked{};cb(cublasLtMatmulAlgoCheck(candidate->handle,candidate->op,candidate->a,candidate->b,candidate->c,candidate->c,&a,&checked),"exact config check");need(checked.state==CUBLAS_STATUS_SUCCESS&&checked.workspaceSize==0,"V301 checked config/workspace mismatch");candidate->algo=a;selected++;
  }
  need(selected==1,"V301 requires one unique screened config; no algorithm substitution");
  }else restored_policy=restore_policy_r2(candidate->algo,device);
  validate_public_r2(candidate->algo);
  cublasLtMatmulHeuristicResult_t restored_checked{};cb(cublasLtMatmulAlgoCheck(candidate->handle,candidate->op,candidate->a,candidate->b,candidate->c,candidate->c,&candidate->algo,&restored_checked),"R2 exact saved-object layout check");need(restored_checked.state==CUBLAS_STATUS_SUCCESS&&restored_checked.workspaceSize==0,"V301 R2 restored config/layout/workspace mismatch");
  candidate->alignment={cap<uint32_t>(candidate->algo,CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES),cap<uint32_t>(candidate->algo,CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES),cap<uint32_t>(candidate->algo,CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES),cap<uint32_t>(candidate->algo,CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES)};
  for(auto n:candidate->alignment)need(n&&n<=256&&256%n==0,"V301 unsupported algorithm alignment");
  if(!component_qualification)need(restored_policy==candidate->policy(),"V301 R2 exact restored policy identity differs; no numerical writes");
  s_=std::move(candidate);if(!rebind(scratch,elems,workspace,bytes,err)){s_.reset();return false;}
  s_->ready=true;std::fprintf(stderr,"V301 PREPARED CUDA%d runtime=%d Lt=%zu id5 tile20 stages7 split1 reduction0 numeric0x10202 ws0 config=",device,runtime,cublasLtGetVersion());for(uint64_t word:s_->algo.data)std::fprintf(stderr,"%016llx",(unsigned long long)word);
  std::fprintf(stderr," custom=%d swizzle=%u scratch=%p/%lld workspace=%p/%zu outside_capture=1 no_tensor_allocation=1 component_qualification=%d policy_bound=%d exact_saved_restore_r2=1\n",config<int>(s_->algo,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION),config<unsigned>(s_->algo,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING),(void*)scratch,(long long)elems,workspace,bytes,int(component_qualification),int(!component_qualification));return true;
 }catch(const std::exception&e){err=std::string("V301 init: ")+e.what();s_.reset();return false;}
}
bool Context::rebind(uint16_t* scratch,int64_t elems,void* workspace,size_t bytes,std::string& err){
 try{need(bool(s_),"V301 rebind without initialized state");int current=-1;ck(cudaGetDevice(&current),"rebind device");need(current==s_->device,"V301 rebind wrong owner");cudaStreamCaptureStatus capture;ck(cudaStreamIsCapturing(s_->stream,&capture),"rebind capture");need(capture==cudaStreamCaptureStatusNone,"V301 rebind inside capture");
  // Caller must already have drained the old complete Prefill lease. Initial
  // initialize has no old numerical users. No borrowed memory is touched here.
  need(aligned(scratch,256)&&elems==ScratchElems&&aligned(workspace,256)&&bytes==WorkspaceBytes,"V301 existing scratch/workspace extent or alignment mismatch");s_->scratch=scratch;s_->elems=elems;s_->workspace=workspace;s_->bytes=bytes;return true;
 }catch(const std::exception&e){err=std::string("V301 rebind: ")+e.what();return false;}
}
Outcome Context::run(strata::prefill::Gemm&gm,const strata::core::WeightRef*w,const uint16_t*x,float*y,int64_t rows,int64_t ldy,int64_t ldx,int kind,std::string&err){
 const int bin=kind==0?0:1;
 // Complete fallback before any CUDA query/dequant/submission. In particular
 // resident protectedBF16 has native_data==nullptr and retains native_proj.
 if(!s_||!s_->ready||!w||kind<0||kind>1||rows!=T||w->ne0!=K||w->ne1!=N||w->kind!=strata::core::WeightKind::Verbatim||w->q8_scale||!w->native_data||(w->native_type!=12&&w->native_type!=14)||(ldy&&ldy!=N)||(ldx&&ldx!=K)||!aligned(x,256)||!aligned(y,256)){
  if(s_)s_->fallbacks[bin]++;return Outcome::fallback;
 }
 try{
  int current=-1;ck(cudaGetDevice(&current),"run owner");need(current==s_->device&&gm.stream()==s_->stream&&gm.scratch()==s_->scratch&&gm.scratch_elems()==s_->elems,"V301 stale/foreign owner or workspace binding");
  need(s_->workspace&&s_->bytes==WorkspaceBytes,"V301 lost workspace lease");
  strata::kernels::dequant_f16(w->native_type,w->native_data,0,N,K,s_->scratch,s_->stream);
  ck(cudaPeekAtLastError(),"original dequant_f16");const float alpha=1.f,beta=0.f;
  cb(cublasLtMatmul(s_->handle,s_->op,&alpha,s_->scratch,s_->a,x,s_->b,&beta,y,s_->c,y,s_->c,&s_->algo,s_->workspace,s_->bytes,s_->stream),"screened FP16xFP16 FP32 output");
  ck(cudaPeekAtLastError(),"Lt enqueue");if(!s_->calls[bin]++)std::fprintf(stderr,"V301 ADMITTED CUDA%d kind=%s T32768 N2560 K6144 nativeType%d config5/20/7/1/0 original_dequant=1 original_workspace=1\n",s_->device,bin?"QSAout":"GDNout",w->native_type);return Outcome::applied;
 }catch(const std::exception&e){s_->errors++;err=std::string("V301 submitted/invariant failure; no fallback: ")+e.what();abort_capture_and_drain(s_->stream);return Outcome::error;}
}
bool Context::export_qualified_policy(const std::string&path,std::string&err)const{
 try{need(s_&&s_->ready&&s_->component_qualification&&!s_->errors&&(s_->calls[0]+s_->calls[1])>0,"V301 policy export requires successful actual component calls and caller byte-proof");
  const std::string line=s_->policy();int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);need(fd>=0,"V301 policy output must be fresh writable file");const ssize_t n=::write(fd,line.data(),line.size());const int close_status=::close(fd);need(n==ssize_t(line.size())&&close_status==0,"V301 policy write failed");return true;
 }catch(const std::exception&e){err=std::string("V301 policy export: ")+e.what();return false;}
}
}
