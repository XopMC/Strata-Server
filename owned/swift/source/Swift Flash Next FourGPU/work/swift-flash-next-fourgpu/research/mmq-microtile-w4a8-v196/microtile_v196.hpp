#pragma once
#include "strata/prefill/moe_mmq.hpp"
namespace microtile_v196 {
enum class Arm { Extracted=1, Microtile=2 };
// Private only; no automatic global/public dispatch replacement. Root owns device,
// stream, initialization and a nonconcurrent caller scratch lease through final drain.
class Context {
 void*impl_=nullptr;
public:
 Context();~Context();Context(const Context&)=delete;Context&operator=(const Context&)=delete;
 void bind_after_drain(void* scratch,size_t bytes,void* stream);
 void run(const strata::prefill::mmq::Product&,void* stream,Arm=Arm::Extracted);
};
}
