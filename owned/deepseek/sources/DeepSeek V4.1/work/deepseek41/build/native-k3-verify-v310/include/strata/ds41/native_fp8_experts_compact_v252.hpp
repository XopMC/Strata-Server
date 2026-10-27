#pragma once
#include <cstdint>
#include <memory>
#include "strata/ds41/native_prefill.hpp"
namespace strata::ds41 {
struct NativeDeviceCompactV252 {const uint8_t*data;size_t bytes;int bits;};
class NativeFp8Experts {
 struct Impl;std::unique_ptr<Impl>impl_;
public:
 explicit NativeFp8Experts(int batch);
 ~NativeFp8Experts();
 void weights(const uint16_t*decoded);
 void packed_weights(NativeDeviceProjection gate,NativeDeviceProjection up,NativeDeviceProjection down);
 void output16_weights_v252(NativeDeviceCompactV252 gate,NativeDeviceCompactV252 up,NativeDeviceCompactV252 down);
 void run(const uint16_t*x,const float*alpha,int rows,uint16_t*out);
};
}
