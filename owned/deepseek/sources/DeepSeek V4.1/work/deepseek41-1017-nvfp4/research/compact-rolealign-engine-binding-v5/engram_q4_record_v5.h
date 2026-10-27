#pragma once
#include <stdint.h>
/* Authoritative typed Engram record contract, separate from expert14/V4.
 * Base row:128 code bytes, low nibble first (2 values/byte), followed by8
 * little-endian BF16 scales (group32). Codes0..14 represent(code-7)*scale;
 * code15 is invalid. No row padding. Base bytes=rows*144.
 * Optional overlay record: I64LE source row ID +256 original FP8E4M3FN bytes
 * +8 original U8 E8M0 scales =272B. IDs sorted unique, within base rows.
 * It preserves original native precision, not BF16 replacement codes.
 * Manifest codec:deepseek41-signed-uniform-bf16-group-v1; encoding15.
 * Segment lists supply contiguous LOGICAL byte concatenation in manifest order;
 * every physical segment must lie in retained owner regions. Metadata is not
 * source authentication; Root's writer/loader must verify source receipts. */
#define DS1017_ENGRAM_Q4_CODEC_V5 "deepseek41-signed-uniform-bf16-group-v1"
#define DS1017_ENGRAM_Q4_ENCODING_V5 15u
typedef struct {uint32_t struct_size,version,encoding,columns;
 uint64_t rows,row_bytes,overlay_rows;
 uint32_t bits,group_size,overlay_row_bytes,reserved;
 uint64_t reserved2;} ds1017_engram_q4_record_v5;
#ifdef __cplusplus
#include <cstddef>
static_assert(sizeof(ds1017_engram_q4_record_v5)==64);
static_assert(offsetof(ds1017_engram_q4_record_v5,rows)==16);
static_assert(offsetof(ds1017_engram_q4_record_v5,bits)==40);
#endif
