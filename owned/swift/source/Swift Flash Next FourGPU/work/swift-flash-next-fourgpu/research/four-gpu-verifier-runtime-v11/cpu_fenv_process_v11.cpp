#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "cpu_fenv_admission_v11.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <dlfcn.h>
#include <elf.h>
#include <gnu/libc-version.h>
#include <link.h>
#include <cstring>
#include <string>
namespace cpu_fenv_admission_v11 {namespace {
struct Find {uintptr_t base;std::string id;};
int note(dl_phdr_info*i,size_t,void*opaque){auto&f=*static_cast<Find*>(opaque);if(uintptr_t(i->dlpi_addr)!=f.base)return 0;for(int j=0;j<i->dlpi_phnum;++j)if(i->dlpi_phdr[j].p_type==PT_NOTE){const auto*at=reinterpret_cast<const unsigned char*>(i->dlpi_addr+i->dlpi_phdr[j].p_vaddr);const auto*end=at+i->dlpi_phdr[j].p_memsz;while(size_t(end-at)>=sizeof(ElfW(Nhdr))){ElfW(Nhdr)h;std::memcpy(&h,at,sizeof h);at+=sizeof h;size_t ns=(h.n_namesz+3u)&~size_t(3),ds=(h.n_descsz+3u)&~size_t(3);if(ns>size_t(end-at)||ds>size_t(end-at)-ns)break;if(h.n_type==NT_GNU_BUILD_ID&&h.n_namesz==4&&!std::memcmp(at,"GNU",4)){char b[3];for(size_t k=0;k<h.n_descsz;++k){std::snprintf(b,sizeof b,"%02x",at[ns+k]);f.id+=b;}return 1;}at+=ns+ds;}}return 1;}
void need(bool x,const char*why){if(!x)throw std::runtime_error(why);}
}
void require_process(){require_ready();need(!std::strcmp(gnu_get_libc_version(),"2.39"),"V11 CPU reference glibc version differs");const char*flag=std::getenv("STRATA_SWIFT_NVFP4_CPU_LEAF_V221");need(flag&&!std::strcmp(flag,"1"),"V11 original CPU-v224 leaf must be selected");for(const char*n:{"STRATA_NO_NVFP4_512","STRATA_FORCE_ISA","STRATA_FORCE_AVX2"})need(!std::getenv(n),"V11 CPU reference ISA override forbidden");need(strata::kernels::cpu::cpu_avx512_ok(),"V11 original CPU AVX512 admission");void*p=dlvsym(RTLD_DEFAULT,"expf","GLIBC_2.27");need(p&&dlsym(RTLD_DEFAULT,"expf")==p,"V11 default/versioned expf mismatch");Dl_info d{};need(dladdr(p,&d)!=0&&uintptr_t(p)-uintptr_t(d.dli_fbase)==0x7dc40,"V11 CPU expf is not pinned __expf_fma");Find f{uintptr_t(d.dli_fbase),{}};dl_iterate_phdr(note,&f);need(f.id=="d53fcbff9f855463606b25cfbe2a2e10856ea3a7","V11 libm build-id differs from qualified CPU oracle");}
}
