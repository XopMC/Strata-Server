#pragma once
#include "resident_gather_v5.hpp"
#include "strata/core/session.hpp"
#include <array>
#include <memory>
namespace resident_pp_v5 {
namespace workspace_detail {struct State;}
// Real independent cudaMalloc ownership. This is never ExpertCache virtual
// bytes/offsets and is never a loan into any immutable physical expert bank.
class WorkspaceLease {
 std::shared_ptr<workspace_detail::State>p_;
 explicit WorkspaceLease(std::shared_ptr<workspace_detail::State>);
 friend class WorkspacePool;
public:
 WorkspaceLease()=default;
 void*base()const;uint64_t bytes()const;int device()const;int rows()const;
 std::shared_ptr<const void>retain()const;
 bool owns(const void*,uint64_t)const;
 // Call AFTER the actual Prefill carve/init. T here is Impl::T allocation, not
 // the request's local min(n,chunk). A last stage passes its terminal capacity.
 AllocatedShape allocated_shape(int actual_T,int T_max,int configured_chunk,int successor_or_terminal_rows)const;
 GroupBuffers group_buffers(unsigned char*gu,unsigned char*dn,float*tail)const;
 void admit_R(const float*actual_R)const;
};
// Declare BEFORE every Prefill/stage resource and its leases, so destruction is
// after all borrowers. allocate() is cold before READY; finish/drain PP before
// any logical update. Pool retirement drains all four contexts before ANY free.
class WorkspacePool {
 std::array<std::shared_ptr<workspace_detail::State>,2>owned_{};
public:
 WorkspacePool()=default;~WorkspacePool();
 WorkspacePool(const WorkspacePool&)=delete;WorkspacePool&operator=(const WorkspacePool&)=delete;
 WorkspaceLease allocate(int dense_owner,Profile,const strata::core::ModelGeometry&,
                         const strata::core::SessionState&stage);
};
} // namespace resident_pp_v5
