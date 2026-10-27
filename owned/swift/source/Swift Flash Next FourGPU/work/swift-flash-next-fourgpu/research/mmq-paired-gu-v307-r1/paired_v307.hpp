#pragma once
#include "strata/prefill/moe_mmq.hpp"
namespace paired_v307 {
struct Epilogue {
 void* hq=nullptr;const float* tails=nullptr;float* row_sd=nullptr;
 int64_t row0=0,nr=0;
 bool materialize_gu=false;
};
enum class Arm { Extracted=1, Paired=2 };
class Context {
 void*impl_=nullptr;
public:
 Context();~Context();Context(const Context&)=delete;Context&operator=(const Context&)=delete;
 void bind_after_drain(void*,size_t,void*);
 void run(const strata::prefill::mmq::Product&,void*stream,Arm=Arm::Extracted);
 // True means fused producer enqueued; false means no work submitted, caller runs original GU/R56.
 bool run_paired(const strata::prefill::mmq::Product&,const Epilogue&,void*stream);
};
}
