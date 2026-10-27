#pragma once
#include "resident_runtime_v11.hpp"
namespace resident_runtime_v11 {
inline constexpr int LedgerLayersV24=24,LedgerStrideV24=80;
void require_protocol_v24();
void cancel_protocol_v24(const char*reason)noexcept;
// Complete/validate all24 GPU receipts, then one bounded ledger D2H+positive wait.
// No accounting or result acceptance is allowed before this succeeds.
bool complete_device_stage_v24(const void*self,std::string&error);
const int32_t*completed_layer_ids_v24(const void*self,int layer,int T);
void require_deferred_accounting_v24(int layer,int T,const int32_t*ids);
}
