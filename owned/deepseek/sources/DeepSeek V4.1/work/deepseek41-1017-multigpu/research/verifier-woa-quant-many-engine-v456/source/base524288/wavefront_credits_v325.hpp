#pragma once
#include <array>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace strata::ds41::wavefront325 {
// The SAME fixed mailbox protocol is used by the production coordinator and
// CPU-only fixture. Packet owns its immutable host storage; no CUDA operation.
template<class Packet>class Credits {
public:
 struct Claim{int slot=-1;uint64_t generation=0;};
 struct Delivery{Claim claim;int layer=-1;Packet packet;};
private:
 enum State{Free,Claimed,Ready,Taken};
 struct Slot{State state=Free;uint64_t generation=0;};
 std::array<std::array<Slot,2>,2>slots_{};
 std::array<std::array<std::optional<Delivery>,40>,2>ready_{};
 std::array<uint64_t,2>published_{};
 std::mutex mu_;std::condition_variable cv_;uint64_t revision_=0;
 std::exception_ptr failure_;
 void healthy()const{if(failure_)std::rethrow_exception(failure_);}
 static void direction(int d){if(d<0||d>1)throw std::invalid_argument("325 mailbox direction");}
public:
 std::optional<Claim>try_claim(int d){direction(d);std::lock_guard<std::mutex>g(mu_);healthy();for(int i=0;i<2;++i)if(slots_[d][i].state==Free){auto&s=slots_[d][i];s.state=Claimed;++s.generation;++revision_;return Claim{i,s.generation};}return {};}
 uint64_t revision(){std::lock_guard<std::mutex>g(mu_);healthy();return revision_;}
 void wait_progress(uint64_t old){std::unique_lock<std::mutex>g(mu_);cv_.wait(g,[&]{return revision_!=old||bool(failure_);});healthy();}
 void publish(int d,Claim c,int layer,Packet packet){direction(d);std::lock_guard<std::mutex>g(mu_);healthy();if(c.slot<0||c.slot>=2||layer<0||layer>=40)throw std::invalid_argument("325 publication bounds");auto&s=slots_[d][c.slot];const uint64_t bit=uint64_t(1)<<layer;if(s.state!=Claimed||s.generation!=c.generation||(published_[d]&bit)||ready_[d][layer])throw std::logic_error("325 stale/duplicate/unclaimed publication");published_[d]|=bit;s.state=Ready;ready_[d][layer]=Delivery{c,layer,std::move(packet)};++revision_;cv_.notify_all();}
 std::optional<Delivery>try_take(int d,int layer){direction(d);std::lock_guard<std::mutex>g(mu_);healthy();if(layer<0||layer>=40)throw std::invalid_argument("325 take layer");auto&q=ready_[d][layer];if(!q)return{};auto out=std::move(q);q.reset();auto&s=slots_[d][out->claim.slot];if(s.state!=Ready||s.generation!=out->claim.generation)throw std::logic_error("325 publication slot lifetime");s.state=Taken;++revision_;return out;}
 Delivery take(int d,int layer){for(;;){auto old=revision();if(auto q=try_take(d,layer))return std::move(*q);wait_progress(old);}}
 void release(int d,Claim c){direction(d);std::lock_guard<std::mutex>g(mu_);healthy();if(c.slot<0||c.slot>=2)throw std::invalid_argument("325 release slot");auto&s=slots_[d][c.slot];if(s.state!=Taken||s.generation!=c.generation)throw std::logic_error("325 premature/stale/duplicate release");s.state=Free;++revision_;cv_.notify_all();}
 void fail(std::exception_ptr e)noexcept{std::lock_guard<std::mutex>g(mu_);if(!failure_)failure_=e;++revision_;cv_.notify_all();}
 void check(){std::lock_guard<std::mutex>g(mu_);healthy();}
 void require_drained(){std::lock_guard<std::mutex>g(mu_);healthy();for(auto&direction:slots_)for(auto&s:direction)if(s.state!=Free)throw std::logic_error("325 mailbox has live credit");const uint64_t full=(uint64_t(1)<<40)-1;if(published_[0]!=full||published_[1]!=full)throw std::logic_error("325 incomplete 40-layer publication masks");}
};
}
