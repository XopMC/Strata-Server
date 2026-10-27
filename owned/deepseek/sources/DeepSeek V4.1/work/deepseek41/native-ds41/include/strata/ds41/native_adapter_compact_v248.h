#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Every pointer must refer to a real readable allocation. The explicit owner
// callbacks retain these allocations until the last engine/registry is freed.
typedef void (*ds41_owner_callback)(void*);
typedef struct { const uint8_t* data; size_t bytes; } ds41_memory_region;
// These values are part of ABI1, not inferred from a third-party tensor enum.
enum ds41_encoding { DS41_BF16=0, DS41_F32=1, DS41_F16=2, DS41_F8E4M3=3,
    DS41_E8M0=4, DS41_I8=5, DS41_U8=6, DS41_I16=7, DS41_I32=8,
    DS41_I64=9, DS41_MXFP4=10, DS41_UNIFORM=11, DS41_OUTPUT16_COMPACT_V248=12 };
typedef struct {
    const char* name; int encoding; const int64_t* shape; size_t rank;
    const uint8_t* data; size_t bytes; int bits; int protected_native_rows;
} ds41_tensor_view;
int ds41_native_adapter_abi(void);
const char* ds41_native_adapter_error(void);
void* ds41_registry_create(const ds41_memory_region*, size_t, void*, ds41_owner_callback, ds41_owner_callback);
int ds41_registry_add(void*, const ds41_tensor_view*);
// Explicit logical geometry, real whitelisted CPU output16 allocation, no scale
// raw tensor alias. Existing raw-view API intentionally cannot register kind12.
int ds41_registry_add_output16_compact_v248(void*,const char*,const void*,size_t,int,int,int);
int ds41_registry_output16_info_v248(void*,uint64_t*,size_t); // exactly3: schema,count,bytes
int ds41_registry_add_segmented(void*, const ds41_tensor_view*,const ds41_memory_region*,size_t);
void ds41_registry_destroy(void*);
void* ds41_checkpoint_create(void*,const char*);
size_t ds41_checkpoint_dense_bytes(void*);
int ds41_checkpoint_protect_rows(void*,const char*,const char*,const char*,const char*);
void ds41_checkpoint_destroy(void*);
void* ds41_rows_create(void*);
int ds41_rows_protect(void*,const char*,const char*,const char*,const char*);
int ds41_rows_gather(void*,const char*,const int64_t*,size_t,uint16_t*,size_t);
void ds41_rows_destroy(void*);
void* ds41_experts_create(void*, const char*, const char*, int, const int*, size_t, int);
int ds41_experts_add_layer(void*, int, const char*, int, int, int, int, float);
int ds41_experts_run(void*, int, int, const uint16_t*, size_t, const int64_t*, size_t, const float*, size_t, uint16_t*, size_t);
int ds41_experts_worker_rows(void*, uint64_t*, size_t);
void ds41_experts_destroy(void*);
#ifdef __cplusplus
}
#endif
