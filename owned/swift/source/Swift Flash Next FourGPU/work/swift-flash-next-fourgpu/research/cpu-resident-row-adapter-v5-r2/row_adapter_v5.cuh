#pragma once
#include "row_adapter_v5.hpp"
#include <cuda_runtime.h>
namespace cpu_rows_v5 {
// Root admits exactly current0..3, all directed native P2P and physical/logical
// leases before capture/allocation. No allocation, host sync or device selection.
// Error belongs to this step: planner resets it, all later operations only OR it.
cudaError_t plan_async(const Inputs*device_inputs,Plan*device_plan,uint32_t*device_error,cudaStream_t);
// Must follow all bank completion events and legacy mapped-copy/hit zero stage.
// Writes CPU rows only. Never raises the old host flag or touches native hit mask.
cudaError_t scatter_async(const Plan*,const Results*device_results,uint32_t*owner_parts,uint32_t*device_error,cudaStream_t);
}
