#pragma once
#include "../four-gpu-physical-banks-v3/physical_types_v3.hpp"
#include <cstdint>
namespace cpu_rows_v5 {
constexpr int MaxT=8,K=10,Capacity=80,H=2560,Devices=4;
enum Kind : uint32_t { Inactive=0, Native=1, Cpu=2, Invalid=3 };
enum Error : uint32_t { BadInput=1,BadRoute=2,EpochMismatch=4,MissingCoverage=8,BadBounds=16,BadResultStamp=32 };
// Entire input image is immutable until all consumers retire. Device pointer
// lets a replay use a freshly published per-ring image without host count readback.
struct Inputs {
 four_banks_v3::PhysicalDeviceView physical;
 four_banks_v3::LogicalDeviceView logical;
 const int32_t* ids=nullptr; // T*K original device router IDs, row order unchanged
 uint64_t expected_physical_epoch=0,expected_logical_epoch=0,sequence=0;
 int32_t layer=-1,T=0;
};
struct Job {
 four_banks_v3::PhysicalRef physical{};
 uint32_t kind=Inactive;
 int32_t key=-1,token=-1,rank=-1,row=-1;
};
struct Plan {
 Job jobs[Capacity]{};
 uint64_t physical_epoch=0,logical_epoch=0,sequence=0;
 int32_t T=0,layer=-1;
};
// Producer writes stamp only after ALL its active route rows are complete.
// Event/graph dependency and native P2P visibility are still mandatory; no polling.
struct ResultStamp { uint64_t physical_epoch=0,logical_epoch=0,sequence=0; };
struct Results {
 const uint32_t* rows[Devices]{}; // fixed [Capacity][H], same original row index
 const ResultStamp* stamps[Devices]{}; // producer-owned, peer-visible
};
// CPU-only independent fixture oracle, not the production planner.
bool reference_plan(const Inputs&,Plan&,uint32_t& error);
bool reference_scatter(const Plan&,const Results&,uint32_t* owner_parts,uint32_t&error);
}
