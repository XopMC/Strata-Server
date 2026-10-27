#pragma once
// Private CUDA-only experiment. No SDK/public class or numerical ABI change.
#include "strata/prefill/gemm.hpp"
#include "strata/core/weights.hpp"
#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
namespace swift_output_lt_v301 {
bool requested(); // process-constant, absent/0 OFF, literal1 ON, other values error
enum class Outcome { fallback, applied, error };
class Context {
 public:
  Context();~Context();Context(const Context&)=delete;Context& operator=(const Context&)=delete;
  bool initialize(int device,void* stream,uint16_t* scratch,int64_t elems,
                  void* workspace,size_t bytes,std::string& err,bool component_qualification=false);
  bool rebind(uint16_t* scratch,int64_t elems,void* workspace,size_t bytes,std::string& err);
  Outcome run(strata::prefill::Gemm&,const strata::core::WeightRef*,const uint16_t* x,
              float* y,int64_t rows,int64_t ldy,int64_t ldx,int kind,std::string& err);
  bool export_qualified_policy(const std::string& path,std::string& err) const;
 private:
  struct State;std::unique_ptr<State> s_;
};
}
