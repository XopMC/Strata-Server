#pragma once
#include <stdexcept>
namespace ds337 {
// Sameissuer bookkeeping, no atomics/devicecalls and no model arithmetic.
class Protocol {
 enum class Phase{Idle,AttentionWriting,Borrowed,AttentionAcked,FFN};
 Phase phase_=Phase::Idle;int next_=0;
 void require(bool v)const{if(!v)throw std::logic_error("337 early/full layer phase or ACK order");}
public:
 bool borrowed()const noexcept{return phase_==Phase::Borrowed;}
 int next_layer()const noexcept{return next_;}
 void begin_attention(int layer){require(phase_==Phase::Idle&&layer==next_&&layer>=0&&layer<40);phase_=Phase::AttentionWriting;}
 void attention_ready(int layer){require(phase_==Phase::AttentionWriting&&layer==next_);phase_=Phase::Borrowed;}
 void acknowledge(int layer){require(phase_==Phase::Borrowed&&layer==next_);phase_=Phase::AttentionAcked;}
 void enter_ffn(int layer){require(phase_==Phase::AttentionAcked&&layer==next_);phase_=Phase::FFN;}
 void full_layer_enqueued(int layer){require(phase_==Phase::FFN&&layer==next_);phase_=Phase::Idle;++next_;}
 void abort_after_drain()noexcept{phase_=Phase::Idle;}
};
}
