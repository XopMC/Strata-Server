#pragma once
// Root bounded whole-row TP fixture support. Extracted unchanged lifetime/guard
// helpers from private V387 component; no model/PP/adaptation integration.
#include "epilogue_v350.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "peer_matrix.hpp"
#include "run_devices.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <immintrin.h>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace K=strata::kernels;namespace fs=std::filesystem;
extern char**environ;
namespace {
using Bytes=std::vector<unsigned char>;
constexpr size_t Guard=256;constexpr unsigned char Pad=0xa5,Canary=0x5a;
void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
void ck(cudaError_t e,const char*s){if(e!=cudaSuccess)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(e));}
void own(int d){ck(cudaSetDevice(d),"select admitted owner");}
void fatal(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"V387_FATAL_DRAIN %s\n",cudaGetErrorString(e));std::_Exit(2);}}
struct Stream {int device;cudaStream_t s{};explicit Stream(int d):device(d){own(d);ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking),"stream");}void drain(){own(device);ck(cudaStreamSynchronize(s),"positive stream completion");}~Stream(){fatal(cudaSetDevice(device));if(s){fatal(cudaStreamSynchronize(s));fatal(cudaStreamDestroy(s));}}};
struct Buffer {
 Stream&stream;size_t bytes;unsigned char*raw{};
 Buffer(Stream&s,size_t n):stream(s),bytes(n){need(n>0&&n<(512ull<<20),"bounded allocation");own(s.device);ck(cudaMalloc(&raw,n+2*Guard),"allocate bounded buffer");try{ck(cudaMemsetAsync(raw,Canary,n+2*Guard,s.s),"canary init");}catch(...){fatal(cudaStreamSynchronize(s.s));cudaFree(raw);throw;}}
 ~Buffer(){fatal(cudaSetDevice(stream.device));fatal(cudaStreamSynchronize(stream.s));fatal(cudaFree(raw));}
 Buffer(const Buffer&)=delete;template<class T=void>T*p()const{return reinterpret_cast<T*>(raw+Guard);}
 void put(const void*src,size_t n){need(n==bytes,"exact upload extent");own(stream.device);ck(cudaMemcpyAsync(p(),src,n,cudaMemcpyHostToDevice,stream.s),"upload fixture");}
 void poison(){own(stream.device);ck(cudaMemsetAsync(p(),Pad,bytes,stream.s),"poison reusable plane");}
 Bytes read(){Bytes v(bytes+2*Guard);own(stream.device);ck(cudaMemcpyAsync(v.data(),raw,v.size(),cudaMemcpyDeviceToHost,stream.s),"proof readback");stream.drain();for(size_t i=0;i<Guard;++i)need(v[i]==Canary&&v[v.size()-1-i]==Canary,"buffer outer canary");return Bytes(v.begin()+Guard,v.end()-Guard);}
 template<class T>T value(){need(sizeof(T)==bytes,"POD read extent");auto b=read();T v;std::memcpy(&v,b.data(),sizeof v);return v;}
 void guards(){unsigned char h[Guard],t[Guard];own(stream.device);ck(cudaMemcpyAsync(h,raw,Guard,cudaMemcpyDeviceToHost,stream.s),"front guard");ck(cudaMemcpyAsync(t,raw+Guard+bytes,Guard,cudaMemcpyDeviceToHost,stream.s),"back guard");stream.drain();for(size_t i=0;i<Guard;++i)need(h[i]==Canary&&t[i]==Canary,"outer guard modified");}
};
struct HostRows {
 float*host{},*device{};size_t bytes;int owner;
 HostRows(int d):bytes(80*2560*4),owner(d){own(d);ck(cudaHostAlloc(reinterpret_cast<void**>(&host),bytes,cudaHostAllocPortable|cudaHostAllocMapped),"mapped original CPU rows");try{ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device),host,0),"owner mapped CPU alias");}catch(...){cudaFreeHost(host);throw;}}
 ~HostRows(){fatal(cudaSetDevice(owner));fatal(cudaFreeHost(host));}
};
struct Graph {
 Stream&stream;cudaGraph_t graph{};cudaGraphExec_t exec{};
 Graph(Stream&s,const std::function<void()>&f):stream(s){own(s.device);bool capturing=false;try{ck(cudaStreamBeginCapture(s.s,cudaStreamCaptureModeThreadLocal),"capture begin");capturing=true;f();ck(cudaGetLastError(),"captured kernels enqueue");auto e=cudaStreamEndCapture(s.s,&graph);capturing=false;ck(e,"capture end");ck(cudaGraphInstantiate(&exec,graph,0),"instantiate source graph");}catch(...){if(capturing){cudaGraph_t bad{};cudaStreamEndCapture(s.s,&bad);if(bad)cudaGraphDestroy(bad);}cudaStreamSynchronize(s.s);if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);throw;}}
 ~Graph(){fatal(cudaSetDevice(stream.device));fatal(cudaStreamSynchronize(stream.s));if(exec)fatal(cudaGraphExecDestroy(exec));if(graph)fatal(cudaGraphDestroy(graph));}
 void launch(){own(stream.device);ck(cudaGraphLaunch(exec,stream.s),"launch reference graph");}
 void admit(){own(stream.device);size_t n=0;ck(cudaGraphGetNodes(graph,nullptr,&n),"source nodes count");std::vector<cudaGraphNode_t>nodes(n);ck(cudaGraphGetNodes(graph,nodes.data(),&n),"source nodes");for(auto node:nodes){cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type),"node type");need(type==cudaGraphNodeTypeKernel,"component child must be kernel-only");cudaGraphNodeParams p{};ck(cudaGraphNodeGetParams(node,&p),"actual kernel context");int d=-1;need(p.kernel.ctx!=nullptr,"kernel context present");ck(cudaExecutionCtxGetDevice(&d,p.kernel.ctx),"actual context ordinal");need(d==stream.device,"captured kernel wrong physical owner");}}
};
struct Parent {
 Stream&stream;cudaGraph_t graph{};cudaGraphExec_t exec{};
 Parent(Stream&s,Graph&pre,Graph&a,Graph&b,Graph&join):stream(s){own(s.device);ck(cudaGraphCreate(&graph,0),"parent create");try{cudaGraphNode_t n[4]{};ck(cudaGraphAddChildGraphNode(&n[0],graph,nullptr,0,pre.graph),"parent pre");ck(cudaGraphAddChildGraphNode(&n[1],graph,&n[0],1,a.graph),"parent branch0");ck(cudaGraphAddChildGraphNode(&n[2],graph,&n[0],1,b.graph),"parent branch1");ck(cudaGraphAddChildGraphNode(&n[3],graph,&n[1],2,join.graph),"parent full join");ck(cudaGraphInstantiate(&exec,graph,0),"parent instantiate");}catch(...){if(exec)cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);throw;}}
 ~Parent(){fatal(cudaSetDevice(stream.device));fatal(cudaStreamSynchronize(stream.s));if(exec)fatal(cudaGraphExecDestroy(exec));if(graph)fatal(cudaGraphDestroy(graph));}
 void launch(){own(stream.device);ck(cudaGraphLaunch(exec,stream.s),"single joined parent launch");}
};
void equal(const Bytes&a,const Bytes&b,const char*label){need(a.size()==b.size(),"comparison extent");auto it=std::mismatch(a.begin(),a.end(),b.begin());if(it.first!=a.end()){std::fprintf(stderr,"V387_DIFF %s byte=%zu expected=%02x actual=%02x\n",label,size_t(it.first-a.begin()),*it.first,*it.second);throw std::runtime_error("exact bytes differ");}}

} // anonymous fixture helpers
