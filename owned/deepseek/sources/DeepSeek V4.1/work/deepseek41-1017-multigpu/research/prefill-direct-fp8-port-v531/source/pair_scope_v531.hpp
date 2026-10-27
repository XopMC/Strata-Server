#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
// No existing class/header is defined/included here. Explicit typed host-only
// call scope is required; an environment flag alone never identifies bytes.
namespace strata::ds41::pair531 {
struct Source {const uint8_t*data=nullptr;size_t bytes=0;int bits=0;};
struct Destination {uint8_t*fp8=nullptr;float*row_scales=nullptr;int rows=0,columns=0;};
struct Inputs {
 uint32_t layout_tag=139,reserved=0;
 uint64_t owner_generation=0,bank_format_epoch=0;
 int expected_device=-1;uint64_t stream=0;
 std::array<Source,3>source{};
};
bool enabled();
class Scope final {
public:
 Scope(const void*instance,const Inputs&inputs);
 ~Scope()noexcept;
 Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
 // Explicit success check before any FFN follows. A wrong/old linked method
 // cannot silently consume a paired image via original244.
 void require_consumed()const;
private:
 friend bool encode_if_scoped(const void*,const std::array<Source,3>&,const std::array<Destination,3>&);
 const void*instance_=nullptr;Inputs inputs_{};int pid_=0;bool consumed_=false;
};
// Called only by private NativeFp8 method implementation; old unscoped callers
// always returnfalse and continue literal244. Wrong-thread same-instance calls
// while a typed scope is active fail instead of interpreting paired as old16.
bool encode_if_scoped(const void*instance,const std::array<Source,3>&source,const std::array<Destination,3>&dest);
}

extern "C" int ds531_prefill_direct_info(int expected_device,uint64_t*out,size_t words);
