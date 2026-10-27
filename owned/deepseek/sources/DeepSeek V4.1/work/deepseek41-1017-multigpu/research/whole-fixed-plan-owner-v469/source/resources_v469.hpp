#pragma once
#include "fixed_plan_info_v469.h"
#include "pair_registry_v145.hpp"
#include "../../whole-adaptive-owner-v370/source/whole_owner_v370.hpp"
#include "../../whole-expert-async-operator-v303/source/whole_operator_v303.hpp"
#include <array>
#include <exception>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
namespace ds469 {
using Triple=std::array<Ds143PairDescriptor,3>;
using Info=std::array<uint64_t,InfoWords>;
struct Closing final:std::exception{const char*what()const noexcept override{return "469 owner closing";}};
struct Busy final:std::exception{const char*what()const noexcept override{return "469 owner busy";}};
class Resources final {
 enum class State {Unused,Copying,Submitted,Ready};
 struct Job {unsigned candidate=0,victim=0,slot=0,credit=0;uint64_t nonce=0,generation=0,format=0;State state=State::Unused;cudaEvent_t done=nullptr;};
 struct Move {unsigned candidate,victim;};
 std::shared_ptr<const strata::ds41::BorrowedWeights>weights_;
 std::vector<Triple>catalog_;std::vector<unsigned>prompt_rank_;
 std::array<int,15360>map_{};std::array<int,1078>slot_id_{};std::array<bool,1078>spare_{};
 std::unordered_map<uintptr_t,unsigned>gate_id_;
 std::array<Job,8>jobs_{};std::vector<Move>moves_;
 std::unique_ptr<ds303::Operator>op_;std::array<uint64_t,ds303::StatCount>last_op_{},reported_op_{};bool reported_finish_=false;uint64_t free_after_workspace_=0;
 std::vector<uint16_t>result_=std::vector<uint16_t>(36*5120);std::array<bool,36>route_selected_{};
 uint8_t *bank_=nullptr,*scratch_=nullptr,*pins_=nullptr;const uint8_t **gu_=nullptr,**down_=nullptr;
 cudaStream_t compute_=nullptr,upload_=nullptr;std::thread worker_;
 mutable std::mutex mutex_;std::condition_variable cv_;
 bool initialized_=false,stop_=false,plan_built_=false;std::atomic<bool>sealed_{false};
 const bool mode_;
 std::atomic<bool>poisoned_{false},closed_{false},closing_{false};std::atomic<uint64_t>hook_refs_{0};
 uint64_t generation_=0,format_=0,issuer_=0,last_issuer_=0,next_nonce_=1,issuer_rebinds_=0;
 std::atomic<uint64_t>active_epoch_{0};std::atomic<bool>task_active_{false};
 unsigned next_move_=0;std::atomic<bool>borrowed_{false};
 Info info_{};
 // Sole Task254 owns training until sealed. Worker reads only AFTER release
 // publication under mutex and these arrays are NEVER modified after seal.
 std::array<uint32_t,15360>score_{},groups_{},routes_{};
 std::array<uint8_t,384>pending_count_{};std::array<uint8_t,15360>pending_forward_{};
 std::atomic<bool>armed_{false};bool observing_=false;unsigned layer_cursor_=0,current_m_=0;
 uint64_t observer_tid_=0,forward_tid_=0;unsigned completed_=0;
 void worker_main()noexcept;void worker_upload(unsigned,unsigned);void worker_drain()noexcept;void build_plan();
 void check_idle()const;unsigned id_for(int,uintptr_t,uintptr_t,uintptr_t)const;
 void publish_ready(unsigned,const uint8_t*);void close_impl();
public:
 static constexpr size_t Matrix=6635520,TripleBytes=3*Matrix,PhysicalSlots=1078,ActiveSlots=1070,PinnedBytes=2*TripleBytes;
 static constexpr size_t DeviceBytes=PhysicalSlots*(TripleBytes+2*sizeof(void*))+Matrix+ds303::Operator::workspace_device_bytes();
 Resources(std::shared_ptr<const strata::ds41::BorrowedWeights>,std::vector<Triple>,std::vector<unsigned>,const Ds276ExpTable&,bool);
 ~Resources()noexcept;
 Resources(const Resources&)=delete;Resources&operator=(const Resources&)=delete;
 void arm(uint64_t,uint64_t);
 void poison()noexcept;
 void before(uintptr_t,unsigned,const uint8_t*,unsigned);
 void after(int);
 int contains(int,uintptr_t,uintptr_t,uintptr_t)const;
 void begin(const DsWholeCommandV299*,void**);void finish(void*,uint64_t,const uint16_t**,uint64_t);
 void release(void*,uint64_t);void abort(void*,uint64_t);
 Info info()const;ds370::Receipt370 legacy()const;
 void hook_refs(uint64_t n)noexcept{hook_refs_.store(n);}
 void close();
};
// Weak registry only supplies diagnostic snapshots. Strong lifetime is owned by
// Owner370 -> Resources and real Handle/Lease/Task hooks. No raw layout access.
void register_diagnostic(const std::shared_ptr<Resources>&);
void unregister_diagnostic(const Resources*)noexcept;
}
