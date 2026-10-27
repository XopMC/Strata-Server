#pragma once
#include "../../../checkpoint_access_v5.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::ds41 {
// Private whole-DSO facade. Pack MUST be rebuilt with this header; never reuse
// old inline NativeCheckpoint layout or cast an old ABI2 checkpoint into it.
class NativeCheckpoint {
 std::shared_ptr<ds1017_rolealign_binding_v5::Checkpoint>compact_;
public:
 explicit NativeCheckpoint(std::shared_ptr<ds1017_rolealign_binding_v5::Checkpoint>c):compact_(std::move(c)){
  const char*f=std::getenv("DS1017_COMPACT_ROLEALIGN_ENGINE_V5");
  if(!compact_||!compact_->production()||!f||std::strcmp(f,"1"))throw std::invalid_argument("production typed V4 checkpoint and explicit engineV5 required BEFORE Pack upload");
 }
 static void validate_geometry(const std::string&v){ds1017_rolealign_binding_v5::Checkpoint::validate_geometry(v);}
 const std::vector<std::string>&dense_names()const{return compact_->dense_names();}
 size_t dense_arena_bytes()const{return compact_->dense_arena_bytes();}
 std::shared_ptr<const BorrowedWeights>weights()const{return compact_->weights()->raw();}
 std::shared_ptr<ds1017_rolealign_binding_v5::Rows>rows()const{return compact_->rows();}
 std::shared_ptr<ds1017_rolealign_binding_v5::Checkpoint>compact_checkpoint()const{return compact_;}
};
}
