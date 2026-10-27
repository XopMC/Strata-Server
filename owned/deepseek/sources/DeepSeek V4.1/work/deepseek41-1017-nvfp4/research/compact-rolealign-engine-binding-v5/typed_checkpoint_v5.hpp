#pragma once
#include "typed_registry_v5.hpp"
#include "strata/ds41/borrowed_rows.hpp"
namespace ds1017_rolealign_binding_v5 {
using ds1017_compact_rolealign_checkpoint_v4::enabled;
using ds1017_compact_rolealign_checkpoint_v4::fixture_enabled;
inline constexpr char inference_revision[]="1bae0e1088993b9b82de08f59de5be5a34e42170";
inline constexpr char heretic_final_sha256[]="14182df29f072a8cc4f730b53c550f73992d8a5f5899cf0ccaa1dbd27c38f9d4";
struct SourceIdentity {std::string source_revision,heretic_final;};
// Identity strings are mandatory manifest binding, not independent file/SHA
// authentication. Root's future canonical loader must verify actual receipts.
class Checkpoint {
public:
 enum class Inventory{Production40x384,Fixture2x2};
 Checkpoint(std::shared_ptr<const Registry>,const std::string&geometry,const SourceIdentity&,Inventory=Inventory::Production40x384);
 static void validate_geometry(const std::string&);
 const std::vector<std::string>&dense_names()const{return dense_;}
 size_t dense_arena_bytes()const{return dense_bytes_;}
 std::shared_ptr<const Registry>weights()const{return weights_;}
 std::shared_ptr<Rows>rows()const{return rows_;}
 Inventory inventory()const noexcept{return inventory_;}
 bool production()const noexcept{return inventory_==Inventory::Production40x384;}
private:
 std::shared_ptr<const Registry>weights_;std::shared_ptr<Rows>rows_;
 std::vector<std::string>dense_;size_t dense_bytes_=0;Inventory inventory_;
};
} // namespace ds1017_rolealign_binding_v5
