#pragma once
#include <stddef.h>
#include <stdint.h>
#include "engram_q4_record_v5.h"
#ifdef __cplusplus
#include "../compact-scale-expert-rolealign-v4/compact_projection_v4.hpp"
typedef ds1017_compact_rolealign_v4::CompactProjectionView ds1017_rolealign_projection_v5;
extern "C" {
#else
typedef struct {const uint8_t*data;uint64_t bytes,row_bytes;uint32_t rows,columns;uint64_t owner_generation;
 uint32_t struct_size,version;uint8_t base_e8,global_e8,scale_delta_bits,max_scale_delta;uint32_t reserved;uint64_t reserved2;} ds1017_rolealign_projection_v5;
#endif
typedef struct ds1017_rolealign_registry_v5 ds1017_rolealign_registry_v5;
typedef struct ds1017_rolealign_checkpoint_v5 ds1017_rolealign_checkpoint_v5;
typedef struct ds1017_rolealign_cpu_v5 ds1017_rolealign_cpu_v5;
typedef void (*ds1017_rolealign_owner_callback_v5)(void*);
typedef struct {const uint8_t*data;size_t bytes;} ds1017_rolealign_region_v5;
typedef struct {const char*name;int encoding;const int64_t*shape;size_t rank;const uint8_t*data;size_t bytes;int bits,protected_native_rows;} ds1017_rolealign_raw_v5;
typedef struct {uint32_t struct_size,abi_version,expert_encoding,codec_version;uint64_t cpu_mode_mask;char codec_id[64];uint64_t supported_encoding_mask;} ds1017_rolealign_capabilities_v5;
/* Separate ABI5, NOT old ds41 ABI2. Callback owner is retained once, then
 * shared by all descendants. Buffers are immutable until its release. Callbacks
 * must not throw. Destroy only after caller threads have joined. */
int ds1017_rolealign_abi_v5(void);
int ds1017_rolealign_capabilities_get_v5(ds1017_rolealign_capabilities_v5*,size_t);
const char*ds1017_rolealign_error_v5(void);
ds1017_rolealign_registry_v5*ds1017_rolealign_registry_create_v5(const ds1017_rolealign_region_v5*,size_t,uint64_t generation,void*token,ds1017_rolealign_owner_callback_v5 retain,ds1017_rolealign_owner_callback_v5 release);
int ds1017_rolealign_registry_add_raw_v5(ds1017_rolealign_registry_v5*,const ds1017_rolealign_raw_v5*,const ds1017_rolealign_region_v5*segments,size_t segment_count);
int ds1017_rolealign_registry_add_compact_v5(ds1017_rolealign_registry_v5*,const char*name,uint32_t encoding,const char*codec,const ds1017_rolealign_projection_v5*);
/* Typed15 only; base/overlay segment arrays concatenate logical record bytes. */
int ds1017_rolealign_registry_add_engram_v5(ds1017_rolealign_registry_v5*,const char*name,uint32_t encoding,const char*codec,const ds1017_engram_q4_record_v5*,const ds1017_rolealign_region_v5*base,size_t base_count,const ds1017_rolealign_region_v5*overlay,size_t overlay_count);
const char*ds1017_rolealign_engram_codec_v5(void);
void ds1017_rolealign_registry_destroy_v5(ds1017_rolealign_registry_v5*);
/* fixture=0 requires full40x384 inventory; fixture=1 explicitly2x2 only. */
ds1017_rolealign_checkpoint_v5*ds1017_rolealign_checkpoint_create_v5(ds1017_rolealign_registry_v5*,const char*geometry,const char*revision,const char*heretic_final,int fixture);
size_t ds1017_rolealign_checkpoint_dense_bytes_v5(ds1017_rolealign_checkpoint_v5*);
int ds1017_rolealign_rows_gather_v5(ds1017_rolealign_checkpoint_v5*,const char*name,const int64_t*,size_t,uint16_t*,size_t);
void ds1017_rolealign_checkpoint_destroy_v5(ds1017_rolealign_checkpoint_v5*);
/* Only explicit ExactIntK32 mode2. No old A8/task arithmetic dispatch. */
ds1017_rolealign_cpu_v5*ds1017_rolealign_cpu_create_v5(ds1017_rolealign_checkpoint_v5*,const char*pool,const int*cpus,size_t workers,int coordinator,int mode);
int ds1017_rolealign_cpu_add_layer_v5(ds1017_rolealign_cpu_v5*,int,const char*,int,int,int,int,float);
int ds1017_rolealign_cpu_prepare_v5(ds1017_rolealign_cpu_v5*,int);
int ds1017_rolealign_cpu_run_selected_v5(ds1017_rolealign_cpu_v5*,int layer,int rows,int topk,const uint16_t*,size_t,const int64_t*,size_t,const float*,size_t,uint16_t*,size_t);
int ds1017_rolealign_cpu_worker_rows_v5(ds1017_rolealign_cpu_v5*,uint64_t*,size_t);
void ds1017_rolealign_cpu_destroy_v5(ds1017_rolealign_cpu_v5*);
#ifdef __cplusplus
}
static_assert(sizeof(void*)==8);
static_assert(sizeof(ds1017_rolealign_projection_v5)==64);
static_assert(sizeof(ds1017_rolealign_capabilities_v5)==96);
#endif
