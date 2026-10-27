#pragma once
#include "fragment_layout.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace row_fixture_v410 {
constexpr int Capacity=80,MaxT=8,Width=2560,Hidden=640,Qrow=2880;
struct Case {
 int T=0,groups=0,entries=0;
 std::array<std::int32_t,80>ids{},dst{},tok{},logical{};
 std::array<std::int32_t,81>start{};
 std::array<std::int32_t,512>resident{};
 std::array<bool,80>native{};
 std::vector<std::uint8_t>q8;
 std::vector<float>cpu,weights,shared;
};
inline Case make_case(int T,int distinct,int mix,int replay){
 if(T<1||T>8||distinct<1||distinct>80||mix<0||mix>2||replay<0||replay>1)throw std::invalid_argument("bounded numeric case");
 Case c;c.T=T;c.ids.fill(-1);c.resident.fill(-1);c.logical.fill(-1);c.q8.resize(8*Qrow,0xa5);c.cpu.resize(80*Width);c.weights.resize(80);c.shared.resize(8*Width);
 for(int e=0;e<distinct;++e)if(mix!=2&&!(mix==1&&e%7==0))c.resident[e]=e;
 for(int t=0;t<T;++t)for(int j=0;j<10;++j)c.ids[t*10+j]=distinct<10?(j<distinct?j:80+j-distinct):(t*10+j+replay)%distinct;
 std::array<bool,512>seen{};
 for(int r=0;r<T*10;++r){int e=c.ids[r];if(seen[e])continue;seen[e]=true;if(c.resident[e]<0)continue;
  c.logical[c.groups]=c.resident[e];c.start[c.groups]=c.entries;
  for(int n=r;n<T*10;++n)if(c.ids[n]==e){c.native[n]=true;c.dst[c.entries]=n;c.tok[c.entries]=n/10;++c.entries;}
  ++c.groups;
 }
 c.start[c.groups]=c.entries;
 for(int t=0;t<T;++t)for(int block=0;block<80;++block){auto*p=c.q8.data()+t*Qrow+block*36;
  const std::uint16_t ds[2]={std::uint16_t(0x1c00+((t+block+replay)%3)*0x400),0};std::memcpy(p,ds,4);
  for(int i=0;i<16;++i){const int v=i==0?127:1+((block*11+i*7+replay*13+t)%63);p[4+2*i]=std::uint8_t(v);p[5+2*i]=std::uint8_t(std::int8_t(-v));}
 }
 for(std::size_t i=0;i<c.cpu.size();++i){float v=float(int((i+replay*19)%37)-18)*0x1p-10f;
  if(replay&&i%131==0)v=std::bit_cast<float>(i%2?0x80000000u:1u);c.cpu[i]=v;
 }
 for(std::size_t i=0;i<c.weights.size();++i)c.weights[i]=float(1+(i+replay)%11)*0x1p-6f;
 for(std::size_t i=0;i<c.shared.size();++i)c.shared[i]=float(int((i+replay)%31)-15)*0x1p-11f;
 return c;
}
inline std::vector<std::uint8_t> synthetic_weights(){
 using namespace row_fragment_v410;std::vector<std::uint8_t>result(80*Blob);
 for(int e=0;e<80;++e){auto*p=result.data()+std::size_t(e)*Blob;
  for(std::size_t at=0;at<Tail;at+=36){const std::uint32_t block=std::uint32_t(at/36);
   for(int k=0;k<4;++k)p[at+k]=std::uint8_t(0x18+(e+k+block)%8);
   for(unsigned k=0;k<32;++k){std::uint32_t v=0x9e3779b9u*(unsigned(e)+1)+0x85ebca6bu*(block+1)+0xc2b2ae35u*(k+1);v^=v>>16;v*=0x7feb352du;v^=v>>15;p[at+4+k]=std::uint8_t(v^(v>>8));}
  }
  const float scales[4]={.5f+float(e%5)*.125f,.25f+float(e%3)*.125f,.125f+float(e%7)*.03125f,0};std::memcpy(p+Tail,scales,16);
 }
 return result;
}
}
