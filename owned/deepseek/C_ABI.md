# Selected native C ABI and build target

The standalone target is **`strata_deepseek_classic`**. Configure exactly one
selected context:

```sh
cmake -S owned/deepseek -B build/deepseek-256 -DSTRATA_DS_CONTEXT=262144
cmake --build build/deepseek-256 --target strata_deepseek_classic

cmake -S owned/deepseek -B build/deepseek-512 -DSTRATA_DS_CONTEXT=524288
cmake --build build/deepseek-512 --target strata_deepseek_classic
```

CUDA 13.4, Clang 18, llvm-ar and mold are required. `STRATA_DS_CONTEXT` is the
native source context choice, not proof of a filled prompt or memory budget.
`STRATA_DS_ENABLE_SERVING_FACTORY` defaults OFF and ON is explicitly refused.
The Python factory and cold launcher/serve entrypoints also carry a fixed false
publication gate. Native source compilation/idle qualification is available;
model inference and public serving are not admitted by this source import.

The result is `<build>/<context>/libds1017_pair_runtime_v1.so` plus newly built
support DSOs in `<build>/support`. Both contexts retain the same original main
SONAME and C API. Load one selected context per fresh GPU child; do not load
both into an interposable namespace.

## Authoritative declarations

Let `M` denote this package's
`sources/DeepSeek V4.1/work/deepseek41-1017-multigpu/research`.

The actual selected `native_engine_gu_v501.o` is rebuilt from
`M/index-native-admission-v501/source/native_engine_gu_v387.cpp`. Its first
matching engine C header, using the frozen include order, is:

`M/whole-adaptive-engine-v387/source/include/strata/ds41/native_engine_causal_batch_v310.h`

That header defines `ds41_native_hash` and `ds41_native_engine_options` and
declares the original Engine C entrypoints. Use this header and its matching
`strata/ds41/native_adapter.h`, resolved for this TU at
`sources/DeepSeek V4.1/work/deepseek41/build/native-k3-verify-v310/include/strata/ds41/native_adapter.h`.
Do not duplicate these structs against fresh upstream classes. Key calls are:

```c
const char* ds41_native_engine_error(void);
void* ds41_native_engine_create(void* checkpoint,
    const ds41_native_hash*, const ds41_native_engine_options*);
int ds41_native_engine_prefill(void*, const int32_t* tokens, size_t count,
    int position, int chunk, int batch, int* next,
    float* logits, size_t logit_count, float* nll, size_t nll_count,
    double* times, size_t time_count);
int ds41_native_engine_step(void*, int token, int position, int* next,
    float* logits, size_t logit_count, uint16_t* hidden, size_t hidden_count,
    int32_t* routes, size_t route_count, double* times, size_t time_count);
int ds41_native_engine_verify(void*, const int32_t* tokens, size_t count,
    int position, int32_t* next, float* logits, size_t logit_count,
    double* times, size_t time_count);
int ds41_native_engine_commit(void*, int keep);
void ds41_native_engine_destroy(void*);
```

These are declaration excerpts, not a replacement public struct definition.
The matching source/header closure also supplies capture, prefill mode/shape,
pool, journal and diagnostic operations. Tentative verification requires the original positive
commit/rollback protocol. The destroy path retires adaptive hooks/resources
before releasing the Engine; a source import does not change that lifetime.

The selected context engine includes the following additional C definitions
from `M/pass-upload-async-native-v558/source/serving_counters_v516.inc`:

```cpp
extern "C" int ds528_prepare_request_v1(int context, int N, int pos,
                                       uint32_t flags) noexcept;
extern "C" int ds528_prefill_plan_info(uint64_t* out, int n) noexcept; // n == 16
extern "C" int ds516_serving_prefill_info(uint64_t* out, int n) noexcept; // n == 12
```

Preparation binds the context-specific request geometry before Engine creation.
It must match the loaded native context. The selected generic native contract is
ABI528, source revision558. The asynchronous upload observer is defined in
`pass_upload_async_v558.hpp` as
`extern "C" int ds558_pass_upload_info(uint64_t*, int) noexcept`; its word count
is the literal `pass558::InfoWords` in that same source. Do not infer semantics
or word counts from a historical variant name.

The original checkpoint/borrowed-weight C ABI is declared by the exact
`native_adapter.h` resolved for each TU in `build-manifest.json`. Its backing
owner, full input schemas, leases and model assets still need the explicit
clean-build publication binder. No copied private DSO is the implementation.
