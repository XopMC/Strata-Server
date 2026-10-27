#pragma once
// One authoritative PolicyHooks370 definition; no copied/guessed ABI struct.
#include "../../whole-adaptive-owner-v370/source/whole_owner_v370.hpp"
#include <type_traits>
static_assert(std::is_standard_layout_v<ds370::PolicyHooks370>);
static_assert(sizeof(ds370::PolicyHooks370)==56);
static_assert(sizeof(ds370::Receipt370)==672);
extern "C" int ds386_moe_policy_attach(void*task_engine,const ds370::PolicyHooks370*hooks)noexcept;
