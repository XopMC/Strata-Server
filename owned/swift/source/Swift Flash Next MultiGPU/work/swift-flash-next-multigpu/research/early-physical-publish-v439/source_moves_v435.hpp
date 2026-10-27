#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace source_moves_v435 {
struct Copy { int key=-1, slot=-1; };

// The original logical cache and arithmetic classification remain authoritative.
// Complete authoritative map snapshot required. A caller may enqueue only bank0
// after its positive root retirement; bank1 writes require successor retirement.
// Metadata only: no weight reads, allocation of payload or CUDA calls.
template<class Mapping>
bool build(std::span<const int32_t> resident,
           const std::array<std::size_t,2>& slots, const Mapping& mapping,
           std::array<std::vector<Copy>,2>& result, std::string& error, bool identity_destination=false) {
    result={};
    auto fail=[&](const char* why){result={};error=why;return false;};
    if(resident.size()!=48*512 || slots[0]==0 || slots[1]==0 ||
       slots[0]>24*512 || slots[1]>24*512)
        return fail("V428 invalid complete logical geometry");
    std::array<std::vector<int>,2> keys, used;
    for(int h=0;h<2;++h){
        if(mapping[h].size()!=slots[h])return fail("V428 map length mismatch");
        keys[h].assign(slots[h],-1); used[h].assign(slots[h],0);
    }
    for(int key=0;key<48*512;++key){
        const int h=key/(24*512),s=resident[key];
        if(s< -1 || (s>=0 && std::size_t(s)>=slots[h]))
            return fail("V428 logical slot outside owning cache");
        if(s>=0){
            if(keys[h][s]!=-1)return fail("V428 duplicate logical slot");
            keys[h][s]=key;
        }
    }
    for(int h=0;h<2;++h){
        if(std::find(keys[h].begin(),keys[h].end(),-1)!=keys[h].end())
            return fail("V428 logical cache has a missing source key");
        for(std::size_t s=0;s<slots[h];++s){
            const auto p=mapping[h][s];
            if(p.bank<0 || p.bank>1 || p.slot<0 || std::size_t(p.slot)>=slots[p.bank])
                return fail("V428 physical destination outside owning bank");
            if(++used[p.bank][p.slot]!=1)return fail("V428 duplicate physical destination");
            if(p.bank!=h || std::size_t(p.slot)!=s)
                result[identity_destination?h:p.bank].push_back({keys[h][s],identity_destination?int(s):p.slot});
        }
    }
    for(int b=0;b<2;++b){
        if(std::find(used[b].begin(),used[b].end(),0)!=used[b].end())
            return fail("V428 physical destination has no logical producer");
        std::sort(result[b].begin(),result[b].end(),[](Copy a,Copy b){return a.slot<b.slot;});
    }
    return true;
}
} // namespace source_moves_v435
