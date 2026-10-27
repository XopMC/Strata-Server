#pragma once
#include "bank_loader_v9.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert.hpp"
namespace four_loader_v9 {
// Does not open, load, copy, register or hash weights. Caller already owns the
// admitted original RAM arena. A shared Arena owner keeps its mapping alive
// through all direct/bounced uploads and failure-drain paths.
inline Source from_original_arena(std::shared_ptr<const strata::core::ArenaExpertSource>arena,uint64_t source_epoch){
 if(!arena||!arena->mapped()||arena->blobs()!=B::Keys||!source_epoch)throw std::runtime_error("V9 original full RAM arena is not ready");
 const auto&layout=strata::kernels::cpu::expert_layout();
 if(!layout.native)throw std::runtime_error("V9 original native expert layout required");
 for(int l=0;l<48;++l)if(layout.blob_bytes(l)!=B::BlobBytes)throw std::runtime_error("V9 native40 blob size mismatch");
 Source s;s.immutable_arena_owner=arena;s.source_epoch=source_epoch;s.one_immutable_arena=true;
 s.get=[arena](int key){if(key<0||key>=B::Keys)throw std::runtime_error("V9 arena key bound");
  return SourceSpan{arena->blob_at(key/512,key%512),B::BlobBytes,arena->pinned(key/512,key%512)?HostKind::Pinned:HostKind::Pageable};};
 return s;
}
}
