#pragma once
#include "strata/ds41/borrowed_weights.hpp"
#include "pair_pool_v143.h"
#include <array>
#include <memory>
#include <string>
namespace strata::ds41::pair145 {
// Distinct internal encoding value, not a changed public TensorEncoding enum.
inline constexpr TensorEncoding Encoding=static_cast<TensorEncoding>(139);
inline constexpr size_t MatrixBytes=6635520, TripleBytes=3*MatrixBytes;
// Startup only. Process-latched explicit opt-in; neither probes payload nor CUDA.
bool enabled();
// Registry retains payload through original BorrowedWeights owner. Sidecar has
// only a weak_ptr, so destroying all registries/checkpoints/Engines releases RAM.
void add(const std::shared_ptr<BorrowedWeights>&owner,const std::string&name,const Ds143PairDescriptor&d);
bool contains(const BorrowedWeights&owner,const std::string&prefix);
Ds143PairDescriptor projection(const BorrowedWeights&owner,const std::string&prefix,int rows,int columns);
// [145,139,count,bytes,owner_generation,bank_format_epoch,enabled,0].
std::array<uint64_t,8> info(const BorrowedWeights&owner);
// [145,139,passes,inverse_calls,input_bytes,output_bytes,scratch_bytes,last_cap,last_sub,peak_scratch,0,0].
void record_prefill(const BorrowedWeights&owner,int cap,int sub);
void record_inverse(const BorrowedWeights&owner,uint64_t calls,uint64_t bytes);
std::array<uint64_t,12> upload_info(const BorrowedWeights&owner);
}
extern "C" {
int ds145_registry_add_pair32(void*registry,const char*name,const Ds143PairDescriptor*d);
int ds145_registry_pair32_info(void*registry,uint64_t*out,size_t count);
int ds145_registry_pair32_upload_info(void*registry,uint64_t*out,size_t count);
}
