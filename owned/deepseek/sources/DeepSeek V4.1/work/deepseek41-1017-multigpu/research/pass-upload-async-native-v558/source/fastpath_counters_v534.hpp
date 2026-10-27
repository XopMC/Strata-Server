#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <stdexcept>
namespace strata::ds41::fast534 {
// Request-local observations; errors are sticky until the CUDA child exits.
inline std::array<std::atomic<uint64_t>,32> words{};
inline void add(int i,uint64_t n=1){words.at(i).fetch_add(n);}
inline void set(int i,uint64_t n){words.at(i).store(n);}
struct Call {
 bool success=false;
 Call(uint64_t epoch,int context,int N,int pos,int E){
  if(words[2].load()||words[18].load()||words[17].load()||words[21].load())
   throw std::logic_error("534 active/failed fastpath cannot be rebound");
  for(int i=0;i<32;i++)words[i]=0;
  set(0,534);set(1,32);set(2,1);set(3,4);set(4,context);set(5,N);set(6,pos);set(7,epoch);set(8,E);set(30,1);
 }
 void complete(){if(words[18].load()||words[9].load()!=1||words[19].load()!=1||words[20].load()!=1||!words[26].load()||words[11].load()!=40*words[26].load()||words[10].load()+1!=words[11].load()||!words[14].load()||words[12].load()!=words[14].load()||words[13].load()!=19906560ull*words[14].load()||words[15].load()!=3*words[14].load())throw std::logic_error("534 incomplete lease/fusion work");add(16);success=true;}
 ~Call(){if(!success)add(17);set(2,0);}
 Call(const Call&)=delete;Call&operator=(const Call&)=delete;
};
}
