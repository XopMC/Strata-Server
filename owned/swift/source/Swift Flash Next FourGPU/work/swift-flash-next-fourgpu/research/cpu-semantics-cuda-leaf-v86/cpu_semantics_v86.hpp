#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
namespace swift_cpu_semantics_v86 {
constexpr int H=2560,F=640,MaxT=8;
constexpr std::size_t Blob=2764816,GuRow=1440,DownRow=360,UpOffset=921600,DownOffset=1843200,TailOffset=2764800;
constexpr std::size_t InputQ=2720,HiddenQ=680;
// Private finite-component ABI only. No routing/production adapter exists.
struct Images {
 const std::uint8_t* weights;
 const float* input;
 std::uint8_t* input_q;
 float *gate_raw,*up_raw,*gate_scaled,*up_scaled,*hidden;
 std::uint8_t* hidden_q;
 float *down_raw,*output;
 std::uint32_t* error;
};
cudaError_t exp_probe(const float*input,float*output,int count,cudaStream_t stream);
cudaError_t quant_probe(const float*input,std::uint8_t*output,int blocks,cudaStream_t stream);
cudaError_t dot_probe(const std::uint8_t*weights,std::size_t row_bytes,const std::uint8_t*q,int n,int rows,int T,int lo,int hi,float*out,cudaStream_t stream);
// Capturable full chain, original packed weights -> original CPU-format Q8.
// All memory supplied/prepared by caller. Every nonfinite chain intermediate
// sets error and prevents qualification. Probe APIs separately test edge bits.
cudaError_t chain(const Images&images,int T,cudaStream_t stream);
}
