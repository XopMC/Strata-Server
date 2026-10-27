#pragma once
#include "resident_plan_v5.hpp"
#include "../four-gpu-physical-banks-v3/physical_banks_v3.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <string>
namespace resident_pp_v5 {
// All three spans are the ORIGINAL per-owner group allocations. Their resource
// owner must be retained; this class never allocates a second PP/tensor arena.
struct GroupBuffers {unsigned char*gu=nullptr,*dn=nullptr;float*tail=nullptr;uint64_t gu_bytes=0,dn_bytes=0,tail_bytes=0;std::shared_ptr<const void>owner;};
// Immediate HOST enqueue callback, never cudaLaunchHostFunc. It must enqueue the
// original GU/R027-completion/Hq/DN consumers on the supplied original stream,
// with the SAME n/bounds/max_rows/T*K/identity row mapping and original combine.
using OriginalConsumers=void(*)(const Group&,const GroupBuffers&,cudaStream_t,void*);
class GatherOwner {
 struct Impl;std::unique_ptr<Impl>p_;
public:
 GatherOwner(int dense_owner,cudaStream_t original_stream,AllocatedShape,
             four_banks_v3::PhysicalLease,GroupBuffers);
 ~GatherOwner();
 GatherOwner(const GatherOwner&)=delete;GatherOwner&operator=(const GatherOwner&)=delete;
 // gather + callback are inseparable for borrowed group-buffer reuse. Success
 // means queued, NOT GPU complete. Later groups use that same serial stream.
 bool with_group(const Group&,OriginalConsumers,void*user,std::string&error)noexcept;
 // Positive complete owner-stream credit before arena/banks rebind/destruction.
 bool finish(std::string&error)noexcept;
};
} // namespace resident_pp_v5
