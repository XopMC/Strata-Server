#include <cuda_runtime.h>
#include "strata/core/required_native_peer.hpp"
#include <sstream>
#include <cstdlib>
#include <array>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
constexpr size_t Max=32ull<<20;
struct Device {unsigned char *src=nullptr,*dst=nullptr;std::array<cudaStream_t,3>s{};};
int main(){try{
 std::string why;strata::core::RequiredNativePeerOptions policy;policy.admitted_devices=2;policy.before_model_allocation_and_capture=true;policy.binding=strata::core::NativePeerBinding::FreshUuidSnapshot;
 const char* ids=std::getenv("STRATA_RUN_GPU_INDICES");const char* uuid=std::getenv("STRATA_RUN_GPU_UUIDS");if(!ids||!uuid)throw std::runtime_error("fresh launch snapshot required");std::string v;std::stringstream is(ids),us(uuid);while(std::getline(is,v,','))policy.current_system_indices.push_back(std::stoi(v));while(std::getline(us,v,','))policy.transient_cuda_uuids.push_back(v);strata::core::RequiredNativePeerReceipt admission;if(!strata::core::require_native_peer(policy,admission,why))throw std::runtime_error(why);
 std::array<Device,2>a;
 for(int d=0;d<2;++d){ck(cudaSetDevice(d));ck(cudaMalloc(&a[d].src,Max));ck(cudaMalloc(&a[d].dst,Max*3));ck(cudaMemset(a[d].src,0x11*(d+1),Max));for(auto&s:a[d].s)ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));ck(cudaDeviceSynchronize());}
 auto check=[&](int dst,int src,int lane,size_t n){std::array<unsigned char,64>x{};ck(cudaMemcpy(x.data(),a[dst].dst+lane*Max,32,cudaMemcpyDeviceToHost));ck(cudaMemcpy(x.data()+32,a[dst].dst+lane*Max+n-32,32,cudaMemcpyDeviceToHost));for(auto b:x)if(b!=0x11*(src+1))throw std::runtime_error("peer payload mismatch");};
 for(size_t n:{size_t(256u<<10),Max}){int reps=n==Max?32:256;
  for(int dst=0;dst<2;++dst)for(int src=0;src<2;++src)if(src!=dst){ck(cudaSetDevice(dst));auto s=a[dst].s[0];for(int k=0;k<4;++k)ck(cudaMemcpyPeerAsync(a[dst].dst,dst,a[src].src,src,n,s));ck(cudaStreamSynchronize(s));auto t=std::chrono::steady_clock::now();for(int k=0;k<reps;++k)ck(cudaMemcpyPeerAsync(a[dst].dst,dst,a[src].src,src,n,s));ck(cudaStreamSynchronize(s));double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();check(dst,src,0,n);std::printf("PAIR src=%d dst=%d bytes=%zu reps=%d GBps=%.6f us_per_copy=%.6f\n",src,dst,n,reps,n*reps/dt/1e9,dt*1e6/reps);std::fflush(stdout);}
 }
 unsigned char* host=nullptr;ck(cudaMallocHost((void**)&host,Max));std::fill(host,host+Max,0x55);
 for(int d=0;d<2;++d){ck(cudaSetDevice(d));for(int direction=0;direction<2;++direction){auto stream=a[d].s[0];auto copy=[&](){return direction==0?cudaMemcpyAsync(a[d].dst,host,Max,cudaMemcpyHostToDevice,stream):cudaMemcpyAsync(host,a[d].src,Max,cudaMemcpyDeviceToHost,stream);};for(int k=0;k<4;++k)ck(copy());ck(cudaStreamSynchronize(stream));auto t=std::chrono::steady_clock::now();for(int k=0;k<32;++k)ck(copy());ck(cudaStreamSynchronize(stream));double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();std::printf("HOST device=%d direction=%s bytes=%zu reps=32 GBps=%.6f\n",d,direction==0?"H2D":"D2H",Max,Max*32/dt/1e9);}}
 ck(cudaFreeHost(host));
 for(int d=0;d<2;++d){ck(cudaSetDevice(d));for(auto s:a[d].s){ck(cudaStreamSynchronize(s));ck(cudaStreamDestroy(s));}ck(cudaFree(a[d].dst));ck(cudaFree(a[d].src));}
 std::puts("TWO_PEER_BANDWIDTH_PASS component_copy_engine_only no_model_TPS");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"PEER_BANDWIDTH_FAIL %s\n",e.what());return 1;}}
