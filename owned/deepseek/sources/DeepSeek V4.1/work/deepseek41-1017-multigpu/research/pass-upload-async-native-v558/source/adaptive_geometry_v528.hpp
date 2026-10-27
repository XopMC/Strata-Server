#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::ds41::geometry335 {
// Same original frame and pass representation, now checked against actual rows.
struct Fixed {
    int context,prompt_end,split,cap,B,E;
    constexpr bool is256()const noexcept{return context==262144;}
    constexpr int suffix_tokens()const noexcept{return prompt_end-split;}
    constexpr bool pass(int first,int count,int position,int input_size)const noexcept{
        return input_size==prompt_end&&(first==0||first==split)&&position==first&&
            count==(first?suffix_tokens():split)&&count>0&&count<=cap;
    }
    static constexpr int ratio(int layer){return layer==2||layer==8||layer==14?2:layer==20?1:0;}
    constexpr uint64_t carry_bytes(int layer,bool suffix)const{
        return suffix&&ratio(layer)==2?uint64_t(prompt_end%2)*512*4:0;
    }
    constexpr uint64_t frame_bytes(int layer,bool suffix)const{
        if(layer<0||layer>=40)throw std::invalid_argument("528 frame layer");
        const int r=ratio(layer),n=suffix?suffix_tokens():split;
        return 131072ull+(r?uint64_t(n/r)*1280ull:0)+2*carry_bytes(layer,suffix);
    }
    constexpr uint64_t payload(bool suffix)const{
        uint64_t n=0;for(int l=0;l<40;++l)n+=frame_bytes(l,suffix);return n;
    }
    constexpr uint64_t max_frame(bool suffix)const{
        uint64_t n=0;for(int l=0;l<40;++l)n=std::max(n,frame_bytes(l,suffix));return n;
    }
    constexpr int suffix_first(int count)const noexcept{return count>8192?((count-4096)/B)*B:0;}
};
inline constexpr Fixed Original512{524288,523712,262144,262144,1024,2048};
inline constexpr Fixed New256{262144,261568,131072,131072,2048,4096};
inline Fixed selected(){
    const char*v=std::getenv("DS1017_PREFILL_GEOMETRY_V335");
    if(!v||!std::strcmp(v,"0"))return Original512;
    if(std::strcmp(v,"256"))throw std::invalid_argument("335 fixed geometry must be literal0 or256");
    return New256;
}
inline bool serial(){const char*v=std::getenv("DS1017_PREFILL_SERIAL_TWO_PASS_V335");if(v&&std::strcmp(v,"0")&&std::strcmp(v,"1"))throw std::invalid_argument("335 literal serial opt-in");return v&&!std::strcmp(v,"1");}
}
namespace strata::ds41::adaptive528 {
inline constexpr uint32_t Images=1,Nll=2,Serial=4,KnownFlags=7;
struct Plan {
    uint64_t epoch=0;
    bool armed=false;
    int mode=0,N=0,pos=0;
    uint32_t flags=0;
    geometry335::Fixed g{};
};
inline Plan make(int context,int N,int pos,uint32_t flags){
    if((context!=262144&&context!=524288)||N<1||pos<0||pos>context||N>context-pos||(flags&~KnownFlags))
        throw std::invalid_argument("528 context/rows/position/flags");
    const auto base=context==262144?geometry335::New256:geometry335::Original512;
    Plan p;p.armed=true;p.N=N;p.pos=pos;p.flags=flags;p.g=base;
    p.mode=4;
    // Experimental contiguous occupancy cost band, not a fixture/token whitelist.
    // Two-rank launch/weight-upload duplication dominates the observed 16k case.
    // Preserve exact full256/full512 partition and numerical leaves.
    if(pos||(flags&(Images|Nll))||N<65536){
        if(flags&Serial)throw std::invalid_argument("528 serial requires eligible two-pass text geometry");
        p.g.prompt_end=N;p.g.split=0;p.g.cap=std::min(N,base.cap);p.g.B=std::min(p.g.cap,base.B);
        p.g.E=std::min(base.E,((std::min(N,p.g.cap)+15)/16)*16);
        return p;
    }
    int split=N>=base.cap+128?base.cap:(N/2/128)*128;
    if(split<128||split%128||N-split<128)throw std::logic_error("528 real aligned spans");
    const int largest=std::max(split,N-split);
    const int cap=((largest+127)/128)*128;
    if(cap<1||cap>base.cap||split>cap||N-split>cap)throw std::logic_error("528 bounded scratch geometry");
    p.g={context,N,split,cap,std::min(base.B,cap),base.E};p.mode=(flags&Serial)?3:2;
    // Pinned transport is bounded by the original two-rank arena, including carry.
    const uint64_t frames=2*(p.g.max_frame(false)+p.g.max_frame(true));
    auto profile_max=base;profile_max.prompt_end=context;
    if(frames>2*(profile_max.max_frame(false)+profile_max.max_frame(true))+16384ull)
        throw std::overflow_error("528 pinned geometry exceeds bounded profile arena");
    return p;
}
inline int first_rows(const Plan&p){return (p.mode==2||p.mode==3)?p.g.split:0;}
inline int second_rows(const Plan&p){return (p.mode==2||p.mode==3)?p.N-p.g.split:0;}
}
