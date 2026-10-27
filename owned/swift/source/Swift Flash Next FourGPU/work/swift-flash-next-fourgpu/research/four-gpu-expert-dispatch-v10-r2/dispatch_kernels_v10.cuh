#pragma once
#include "dispatch_v10.hpp"
namespace four_dispatch_v10::detail {
cudaError_t resolve_native(cpu_rows_v5::Plan*,const cpu_rows_v5::Inputs*,uint32_t*,NativeMask*,uint32_t*,int layer,int T,cudaStream_t);
cudaError_t copy_inputs(OwnerView,const BankBuffers&,const BankBuffers&owner,cudaStream_t);
cudaError_t native_plan(const BankBuffers&,int bank,cudaStream_t);
cudaError_t bank_commit(const BankBuffers&,int bank,cudaStream_t);
cudaError_t gather_native(OwnerView,const std::array<BankBuffers,4>&,cudaStream_t);
cudaError_t gather_cpu(OwnerView,const std::array<BankBuffers,4>&,cudaStream_t);
}
