#pragma once
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <climits>
#include <cerrno>
#include <cstring>
#include <new>
#include <vector>
#include <stdexcept>
namespace ds418_memory {
inline void need(bool v,const char*m){if(!v)throw std::runtime_error(m);}
template<class T>class HomeArray {
 T*p=nullptr;size_t n=0,bytes=0;
public:
 HomeArray(size_t count,int node):n(count),bytes((count*sizeof(T)+4095)&~size_t(4095)){
  need(n&&node>=0&&node<2,"home array geometry");void*raw=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);need(raw!=MAP_FAILED,"home metadata mmap");
  unsigned long mask=1ul<<node;if(syscall(SYS_mbind,raw,bytes,MPOL_BIND,&mask,sizeof(mask)*CHAR_BIT,MPOL_MF_STRICT)){
   const int saved=errno;munmap(raw,bytes);throw std::runtime_error("CPU417 home metadata mbind errno="+std::to_string(saved)+" ("+std::strerror(saved)+") node="+std::to_string(node)+" mask="+std::to_string(mask)+" maxnode="+std::to_string(sizeof(mask)*CHAR_BIT)+" flags="+std::to_string(MPOL_MF_STRICT)+" bytes="+std::to_string(bytes)+" address_mod4096="+std::to_string(uintptr_t(raw)%4096));
  }
  p=static_cast<T*>(raw);try{for(size_t i=0;i<n;i++)new(p+i)T{};
   std::vector<void*>pages;for(size_t i=0;i<bytes;i+=4096)pages.push_back(reinterpret_cast<uint8_t*>(p)+i);std::vector<int>status(pages.size(),-1);
   need(!syscall(SYS_move_pages,0,pages.size(),pages.data(),nullptr,status.data(),0),"metadata physical query");for(int home:status)need(home==node,"metadata physical home");
  }catch(...){munmap(p,bytes);p=nullptr;throw;}
 }
 ~HomeArray(){if(p){for(size_t i=0;i<n;i++)p[i].~T();munmap(p,bytes);}}
 HomeArray(const HomeArray&)=delete;HomeArray&operator=(const HomeArray&)=delete;
 T*data(){return p;}T*begin(){return p;}T*end(){return p+n;}size_t size()const{return n;}T&operator[](size_t i){return p[i];}
};

}
