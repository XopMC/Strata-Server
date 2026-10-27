#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
namespace row_transport_v410 {
cudaError_t input(const std::uint8_t*source,std::uint8_t*target,int tokens,cudaStream_t);
cudaError_t hidden(const std::uint8_t*source,std::uint8_t*target,const std::int32_t*counts,
 const std::uint32_t*source_error,std::uint32_t*target_error,int source_bank,int capacity,cudaStream_t);
}
