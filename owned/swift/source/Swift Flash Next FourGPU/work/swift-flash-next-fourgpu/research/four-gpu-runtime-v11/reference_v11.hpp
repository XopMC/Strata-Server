#pragma once
// Independent control metadata only. No expert payload or policy selection.
#include "strata/core/expert_cache.hpp"
#include <cuda_runtime_api.h>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <locale>
#include <span>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace four_reference_v11 {
constexpr size_t Keys=24576;
inline void need(bool b,const char*s){if(!b)throw std::runtime_error(std::string("FOUR11 reference: ")+s);}
inline void ck(cudaError_t x,const char*s){if(x!=cudaSuccess)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(x));}
using Bytes=std::vector<unsigned char>;
inline void put(Bytes&b,uint64_t x,int n){for(int k=0;k<n;++k)b.push_back(static_cast<unsigned char>(x>>(k*8)));}
struct Reader{const Bytes&b;size_t p=0;uint64_t get(int n){need(n>0&&n<=8&&p<=b.size()&&size_t(n)<=b.size()-p,"truncated metadata");uint64_t x=0;for(int k=0;k<n;++k)x|=uint64_t(b[p++])<<(8*k);return x;}};
inline uint64_t fnv(std::span<const unsigned char>b){uint64_t h=0xcbf29ce484222325ULL;for(auto x:b){h^=x;h*=0x100000001b3ULL;}return h;}
inline uint64_t input_hash(std::span<const int64_t>v){Bytes b;b.reserve(v.size()*4);for(auto x:v){need(x>=0&&x<248320,"input ID bound");put(b,uint32_t(x),4);}return fnv(b);}
template<class O>std::string tuple(const O&o){std::ostringstream s;s.imbue(std::locale::classic());s<<std::hexfloat;
 s<<"two24-native40-v11|"<<o.max_context<<'|'<<o.kv<<'|'<<o.kv_resident<<'|'<<o.prefill_chunk<<'|'<<o.spec<<'|'<<o.mtp_max_t<<'|'<<o.suffix_draft<<'|'<<o.spec_min_p<<'|'<<o.lookup_chain<<'|'<<o.adapt_swaps<<'|'<<o.adapt_every<<'|'<<o.adapt_decay<<'|'<<o.adapt_min_gain<<'|'<<o.vram_reserve_mib<<'|'<<o.vram_reserve_later_mib<<'|'<<o.rope_scaling<<'|'<<o.rope_scale<<'|'<<o.rope_freq_base<<'|'<<o.rope_freq_scale<<'|'<<o.yarn_orig_ctx<<'|'<<o.yarn_ext_factor<<'|'<<o.yarn_attn_factor<<'|'<<o.yarn_beta_fast<<'|'<<o.yarn_beta_slow;
 return s.str();}
struct Metadata {
 std::string options;uint32_t rows=0;std::array<uint32_t,2>slots{};std::array<uint64_t,2>bytes{},pp_price{};
 std::vector<int32_t>rank,map;std::array<std::vector<uint64_t>,2>offsets;
 // completed reference is published only after the original full PP and checked
 // refill; candidate binds the same exact first request, never a universal map.
 uint64_t input_count=0,input_fnv=0;bool postprefill=false;
 Bytes encode()const{Bytes b{'R','4','M','E','T','A','1','1'};put(b,1,4);put(b,options.size(),4);b.insert(b.end(),options.begin(),options.end());put(b,rows,4);for(int d=0;d<2;++d){put(b,slots[d],4);put(b,bytes[d],8);put(b,pp_price[d],8);}for(auto*v:{&rank,&map}){put(b,v->size(),4);for(auto x:*v)put(b,uint32_t(x),4);}for(auto&v:offsets){put(b,v.size(),4);for(auto x:v)put(b,x,8);}put(b,input_count,8);put(b,input_fnv,8);put(b,postprefill,4);return b;}
 static Metadata decode(const Bytes&b){need(b.size()>=12&&!std::memcmp(b.data(),"R4META11",8),"schema magic");Reader r{b,8};need(r.get(4)==1,"schema version");Metadata m;auto n=r.get(4);need(n&&n<4096&&n<=b.size()-r.p,"options extent");m.options.assign(reinterpret_cast<const char*>(b.data()+r.p),size_t(n));r.p+=size_t(n);m.rows=uint32_t(r.get(4));need(m.rows==8192||m.rows==16384,"actual PP allocation profile");for(int d=0;d<2;++d){m.slots[d]=uint32_t(r.get(4));m.bytes[d]=r.get(8);m.pp_price[d]=r.get(8);need(m.pp_price[d]>0&&m.pp_price[d]<(1ULL<<40),"original actual PP price");need(m.slots[d]>128&&m.slots[d]<=12288&&m.bytes[d]==uint64_t(m.slots[d])*2765056,"logical capacity/stride");}for(auto*v:{&m.rank,&m.map}){need(r.get(4)==Keys,"whole rank/R extent");v->resize(Keys);for(auto&x:*v)x=int32_t(uint32_t(r.get(4)));}for(int d=0;d<2;++d){need(r.get(4)==uint64_t(m.slots[d])+1,"offset extent");m.offsets[d].resize(size_t(m.slots[d])+1);for(size_t q=0;q<m.offsets[d].size();++q){m.offsets[d][q]=r.get(8);need(m.offsets[d][q]==q*2765056ULL,"exact original slot offset");}}m.input_count=r.get(8);m.input_fnv=r.get(8);auto ready=r.get(4);need(ready<=1&&r.p==b.size(),"completion/trailing bytes");m.postprefill=ready;
 std::vector<unsigned char>seen(Keys);for(auto key:m.rank)need(key>=0&&key<int(Keys)&&!seen[size_t(key)]++,"complete original rank permutation");for(int d=0;d<2;++d){std::vector<unsigned char>used(m.slots[d]);for(size_t key=size_t(d)*12288;key<size_t(d+1)*12288;++key){auto s=m.map[key];need(s>=-1&&s<int32_t(m.slots[d]),"logical slot bound");if(s>=0)need(!used[size_t(s)]++,"duplicate logical slot");}for(auto x:used)need(x==1,"all original logical slots used");}return m;}
 static Metadata load(){const char*p=std::getenv("STRATA_FOUR_REFERENCE_V11");need(p&&*p=='/',"absolute independent reference required");std::ifstream f(p,std::ios::binary|std::ios::ate);need(bool(f),"open independent reference");auto n=f.tellg();need(n>0&&n<=1048576,"bounded reference size");Bytes b(size_t(n),0);f.seekg(0);need(bool(f.read(reinterpret_cast<char*>(b.data()),n)),"read independent reference");auto m=decode(b);need(m.postprefill&&m.input_count>1&&m.input_fnv,"reference lacks checked matched-profile prefill");return m;}
};
inline const Metadata&reference(){static const Metadata m=Metadata::load();return m;}
struct Bank{int device=-1;strata::core::ExpertCache*cache=nullptr;cudaStream_t stream=nullptr;int32_t*d_res=nullptr;};
inline void checked_tables(const std::array<Bank,2>&banks,std::span<const int32_t>R,const Metadata&m){need(R.size()==Keys&&std::equal(R.begin(),R.end(),m.map.begin()),"authoritative R differs from independent selection");int old=-1;ck(cudaGetDevice(&old),"save table context");struct Restore{int d;~Restore(){if(cudaSetDevice(d)!=cudaSuccess)std::_Exit(2);}}restore{old};for(int d=0;d<2;++d){auto&b=banks[d];need(b.device==d&&b.cache&&b.cache->valid()&&b.stream&&b.d_res,"actual home binding");auto&c=*b.cache;need(c.slots()==m.slots[d]&&c.resident()==c.slots()&&c.bytes()==int64_t(m.bytes[d])&&!c.per_layer_admission()&&!c.segmented(),"original logical shape");for(uint32_t q=0;q<=m.slots[d];++q)need(c.slot_offset(q)==m.offsets[d][q],"original logical offsets");for(size_t key=0;key<Keys;++key)need(c.slot_of(key/512,key%512)==(key/12288==size_t(d)?R[key]:-1),"logical key identity");ck(cudaSetDevice(d),"select table context");ck(cudaStreamSynchronize(b.stream),"table original stream drain");std::vector<int32_t>v(Keys);ck(cudaMemcpy(v.data(),b.d_res,Keys*4,cudaMemcpyDeviceToHost),"actual R readback");ck(cudaStreamSynchronize(nullptr),"actual R readback credit");need(std::equal(v.begin(),v.end(),R.begin()),"actual d_res differs from unchanged logical R");}}
inline void write_new(const std::string&p,const Metadata&m){auto b=m.encode();(void)Metadata::decode(b);FILE*f=std::fopen(p.c_str(),"wbx");need(f,"export must be new");bool ok=std::fwrite(b.data(),1,b.size(),f)==b.size();int e=std::fclose(f);need(ok&&!e,"export completion");}
inline void receipt(const Metadata&m,const char*phase){auto b=m.encode();std::fprintf(stderr,"FOUR11_REFERENCE phase=%s rows=%u slots=%u,%u metadata_fnv=%016llx input_count=%llu input_fnv=%016llx\n",phase,m.rows,m.slots[0],m.slots[1],(unsigned long long)fnv(b),(unsigned long long)m.input_count,(unsigned long long)m.input_fnv);}
} // namespace four_reference_v11
