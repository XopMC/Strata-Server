#pragma once
#include "strata/ds41/native_experts.hpp"
#include "../../whole-adaptive-owner-v370/source/whole_owner_v370.hpp"
#include <functional>
#include <mutex>
#include <memory>
namespace ds387 {
bool enabled();
bool adaptive_mode();
class Lease final {
public:
 Lease(strata::ds41::NativeExperts&,std::shared_ptr<ds370::Owner>,std::function<void()>);
 ~Lease()noexcept;
 Lease(const Lease&)=delete;Lease&operator=(const Lease&)=delete;
 void arm(uint64_t,uint64_t);
 unsigned gap(uint64_t,uint64_t,uint64_t,unsigned);
 ds370::Receipt370 receipt()const;
 bool closed()const;
 void close(const strata::ds41::NativeExperts&);
 void mark_registered();
private:
 mutable std::mutex mutex_;
 const strata::ds41::NativeExperts*identity_;
 std::shared_ptr<ds370::Owner>owner_;
 std::function<void()>detach_;
 uint64_t generation_=0,format_=0,nonce_=0,last_serviced_=0;
 bool registered_=false,closed_=false,armed_=false;
};
std::shared_ptr<Lease>install(strata::ds41::NativeExperts&,std::shared_ptr<ds370::Owner>,std::function<void()>);
std::shared_ptr<Lease>lookup(const strata::ds41::NativeExperts&);
void cleanup(const strata::ds41::NativeExperts*)noexcept;
}
