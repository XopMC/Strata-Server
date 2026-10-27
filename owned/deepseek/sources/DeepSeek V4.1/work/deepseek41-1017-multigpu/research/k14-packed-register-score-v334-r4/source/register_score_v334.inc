// V334: same128 keys/fullK128 queries; explicit PTX and opaque BF16-product transpose.
// This file is private to the R4 derivative. No WMMA fragment.x interpretation.
struct __align__(32) ScoreTile334 {
    bf16 keys[128*128];
    bf16 queries[16*4*128];
    bf16 weights[16*4];
};
static_assert(sizeof(ScoreTile334)==49280,"V334 full shared ledger");
static std::array<std::atomic<uint64_t>,16> score_stats334{};
static bool score_flag334(){
    static const bool on=[](){const char*v=std::getenv("DS1017_K14_PACKED_REGISTER_V334");
        if(!v||!std::strcmp(v,"0"))return false;
        if(!std::strcmp(v,"1"))return true;
        throw std::invalid_argument("V334 flag must be literal0/1");}();
    return on;
}
__device__ __forceinline__ int offset334(int row,int col){return row*128+(col^((row&7)*8));}
__device__ __forceinline__ unsigned shared334(const void*p){return unsigned(__cvta_generic_to_shared(p));}
__device__ __forceinline__ void load_a334(unsigned(&a)[4],const bf16*p){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        :"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(shared334(p)):"memory");
}
__device__ __forceinline__ void load_b334(unsigned(&b)[2],const bf16*p){
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        :"=r"(b[0]),"=r"(b[1]):"r"(shared334(p)):"memory");
}
__device__ __forceinline__ void mma334(float(&c)[4],const unsigned(&a)[4],const unsigned(&b)[2]){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ unsigned short product334(float value,bf16 coefficient){
    // Literal original dot RN, fmax, multiply, product RN. Move its original16bits.
    const float dot=__bfloat162float(__float2bfloat16_rn(value));
    const float term=fmaxf(dot,0.0f)*__bfloat162float(coefficient);
    return __bfloat16_as_ushort(__float2bfloat16_rn(term));
}
template<int Bit>
__device__ __forceinline__ void transpose_stage334(unsigned(&v)[4],int s){
#pragma unroll
    for(int r=0;r<4;r++)if(!(r&Bit)){
        const unsigned send=(s&Bit)?v[r]:v[r|Bit];
        const unsigned other=__shfl_xor_sync(0xffffffffu,send,Bit<<2);
        if(s&Bit)v[r]=other;else v[r|Bit]=other;
    }
}
__global__ __launch_bounds__(512,2) void batched_scores334(
        const bf16*q,const bf16*keys,const bf16*w,int first,int rows,int pos0,int ratio,
        bf16*scores,int64_t score_stride){
    extern __shared__ __align__(32) unsigned char memory[];
    auto&tile=*reinterpret_cast<ScoreTile334*>(memory);
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const int tile_row=int(blockIdx.y)*16,tile_rows=min(16,rows-tile_row);
    const int64_t key_base=int64_t(blockIdx.x)*128;
    const int64_t tile_end=(int64_t(pos0)+first+tile_row+tile_rows)/ratio;
    if(key_base>=tile_end)return; // uniform BEFORE any barrier, same original causal cut
    for(int item=tid;item<128*128;item+=512){
        const int r=item/128,d=item%128;const int64_t j=key_base+r;
        tile.keys[offset334(r,d)]=j<tile_end?keys[size_t(j)*128+d]:__float2bfloat16_rn(0.f);
    }
    const int quad=warp>>2,parity=warp&3;
    float total[4]={};
#pragma unroll 1
    for(int hb=0;hb<32;hb+=4){
        for(int item=tid;item<16*4*128;item+=512){
            const int qh=item/128,d=item%128,r=qh/4,h=qh%4;
            tile.queries[offset334(qh,d)]=r<tile_rows?q[(size_t(first+tile_row+r)*32+hb+h)*128+d]:__float2bfloat16_rn(0.f);
        }
        if(tid<64){const int r=tid/4,h=tid%4;tile.weights[tid]=r<tile_rows?w[size_t(first+tile_row+r)*32+hb+h]:__float2bfloat16_rn(0.f);}
        __syncthreads(); // publish full query/weight group and retained keys
        // Independent outputs: finish one 64-key part before making the next
        // part's accumulators live. Each dot keeps its original eight K16 MMAs.
#pragma unroll 1
        for(int part=0;part<2;part++){
            float c[2][4]={};
#pragma unroll
            for(int d=0;d<128;d+=16){
                unsigned a[4];
                load_a334(a,tile.queries+offset334(quad*16+(lane%16),d+8*(lane/16)));
#pragma unroll
                for(int half=0;half<2;half++){
                    unsigned b[2];const int kg=parity+4*part;
                    load_b334(b,tile.keys+offset334(kg*16+half*8+lane%8,d+8*((lane/8)%2)));
                    mma334(c[half],a,b);
                }
            }
            const int s=(lane>>2)&3,qlow=lane>>4;
            unsigned v[4];
#pragma unroll
            for(int qhi=0;qhi<2;qhi++){
                const int query=quad*4+qlow+2*qhi;
                const bf16 coefficient=tile.weights[query*4+s];
#pragma unroll
                for(int half=0;half<2;half++){
                    const unsigned lo=product334(c[half][qhi*2],coefficient);
                    const unsigned hi=product334(c[half][qhi*2+1],coefficient);
                    v[qhi*2+half]=lo|(hi<<16);
                }
            }
            transpose_stage334<1>(v,s);transpose_stage334<2>(v,s);
#pragma unroll
            for(int head=0;head<4;head++){
                total[part*2]+=__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(v[head]&0xffffu)));
                total[part*2+1]+=__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(v[head]>>16)));
            }
        }
        __syncthreads(); // all query/weight readers retired before NEXT full group overwrite
    }
    const int s=(lane>>2)&3,qlow=lane>>4;
    const int local_row=quad*4+qlow+2*(s>>1),row=first+tile_row+local_row;
#pragma unroll
    for(int part=0;part<2;part++){
        const int col=(parity+4*part)*16+8*(s&1)+2*(lane&3);
#pragma unroll
        for(int j=0;j<2;j++){
            const int64_t key=key_base+col+j,n=(int64_t(pos0)+row+1)/ratio;
            if(local_row<tile_rows&&key<n)scores[size_t(tile_row+local_row)*size_t(score_stride)+size_t(key)]=__float2bfloat16_rn(total[part*2+j]);
        }
    }
}
static void record_launch334(int rows,uint64_t tiles,uint64_t keys){
    score_stats334[7].fetch_add(1,std::memory_order_relaxed);
    score_stats334[8].fetch_add(tiles,std::memory_order_relaxed);
    uint64_t old=score_stats334[9].load(std::memory_order_relaxed);
    while(old<keys&&!score_stats334[9].compare_exchange_weak(old,keys,std::memory_order_relaxed)){}
    (void)rows;
}
