#pragma once
#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <vector>
namespace early_credit_v439 {
class Credit {
 enum class Phase {Empty,Submitting,Submitted,Validated,Complete,Failed};
 Phase phase_=Phase::Empty; uint64_t epoch_=0; std::vector<int32_t> map_;
 bool reject(std::string&e,const char*why){phase_=Phase::Failed;e=why;return false;}
public:
 bool reset(std::string&e){if(phase_!=Phase::Empty&&phase_!=Phase::Complete)return reject(e,"V439 unretired previous early publication");phase_=Phase::Empty;epoch_=0;map_.clear();return true;}
 bool begin(uint64_t epoch,std::span<const int32_t>map,std::string&e){if(phase_!=Phase::Empty||!epoch||map.size()!=48*512)return reject(e,"V439 duplicate or invalid early publication");map_.assign(map.begin(),map.end());epoch_=epoch;phase_=Phase::Submitting;return true;}
 bool submitted(std::string&e){if(phase_!=Phase::Submitting)return reject(e,"V439 no complete early enqueue");phase_=Phase::Submitted;return true;}
 bool final(uint64_t epoch,std::span<const int32_t>map,std::string&e){if(phase_!=Phase::Submitted||epoch!=epoch_||map.size()!=map_.size()||!std::equal(map.begin(),map.end(),map_.begin()))return reject(e,"V439 stale/missing early publication or changed restored logical map");phase_=Phase::Validated;return true;}
 bool complete(std::string&e){if(phase_!=Phase::Validated)return reject(e,"V439 completion without final map validation");phase_=Phase::Complete;return true;}
 void failed(){phase_=Phase::Failed;}
};
}
