#pragma once
#include "strata/kernels/qsa_select.hpp"
#include <cstdint>
#include <string>
namespace swift_selector_v7 {
constexpr int MaxQueries=256,MaxStreams=8,Cap=2051;
constexpr int64_t MaxBlocks=262146;
constexpr uint64_t BytesPerQuery=106980,GuardBytes=256;
// Caller owns current CUDA context, streams and graph execs. Prepare BEFORE capture.
bool prepare(const void*owner,void*stream,int max_queries,std::string&err) noexcept;
// AFTER all graph exec destruction; positively drains every prepared stream.
bool release(const void*owner,std::string&err) noexcept;
void release(const void*owner) noexcept; // destructor form: fail closed on cleanup failure
class Scope {
 void*previous_=nullptr;void*current_=nullptr;
public:
 explicit Scope(const void*owner);
 ~Scope() noexcept;
 Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
};
// Same Scope/device until enqueue ends. GPU step records are trusted coupled
// qsa_step_fill records, read again on every replay; no capture-time count freeze.
void select(const float*,const int32_t*,int64_t queries,int64_t score_stride,int64_t cap,
            const strata::kernels::QsaShapes&,int32_t*,void*stream,int64_t active=-1);
inline void topk(const float*s,const int32_t*t,int64_t q,int64_t b,int64_t c,const strata::kernels::QsaShapes&h,int32_t*i,void*st,void*,uint64_t){select(s,t,q,b,c,h,i,st,-1);}
inline void topk(const float*s,const int32_t*t,int64_t q,int64_t b,int64_t c,const strata::kernels::QsaShapes&h,int32_t*i,void*st,int64_t a,void*,uint64_t){select(s,t,q,b,c,h,i,st,a);}
struct PoolInfo {int device=-1,queries=0,slots=0;unsigned active=0;uint64_t payload_bytes=0;bool failed=false,retiring=false;};
bool inspect(const void*owner,PoolInfo&out) noexcept; // host metadata, not replay evidence
unsigned live_pools() noexcept;
}
