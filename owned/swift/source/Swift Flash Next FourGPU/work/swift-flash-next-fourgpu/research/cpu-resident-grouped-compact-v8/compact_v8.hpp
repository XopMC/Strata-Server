#pragma once
#include "../cpu-resident-grouped-chain-v6-r1/grouped_v6.hpp"
namespace cpu_compact_v8 {
struct WorkList {uint64_t sequence=0;uint32_t count=0;int32_t rows[80]{};};
// Exactly the same Images/Status/result-row ABI as qualified V6-r1.
// Caller supplies an admitted bounded grid, e.g. 2*current SM count.
cudaError_t chain(const cpu_grouped_v6::Images&,WorkList*,int bank,int grid_ctas,cudaStream_t);
}
