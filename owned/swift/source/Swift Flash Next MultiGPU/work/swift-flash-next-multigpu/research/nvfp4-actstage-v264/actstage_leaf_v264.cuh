// Private GU40-only existing-Q8 stage. Original Split40.load, DP4A and FP32 tree; no new quantization.
// Included inside the same anonymous namespace as the original modern numeric helpers.
constexpr int AS_NC_V264=4, AS_XB_V264=80;
__device__ __forceinline__ void actstage_copy_v264(int4* su,float* sd,int c,const block_q8_1* src,int tid) {
    int* words=(int*)su;const int* in=(const int*)src;
    for(int w=tid;w<AS_XB_V264*9;w+=256){
        const int k=w/9,j=w-k*9,v=in[w];
        if(j==0){const half2 h=*reinterpret_cast<const half2*>(&v);sd[c*AS_XB_V264+k]=__low2float(h);}
        else {const int q=j-1;words[((c*2+(q>>2))*AS_XB_V264+k)*4+(q&3)]=v;}
    }
}
__device__ __forceinline__ float actstage_apply_v264(const Split<40>::W& r,const int4* su,const float* sd,int c,int k) {
    const int4 a=su[(c*2+0)*AS_XB_V264+k],b=su[(c*2+1)*AS_XB_V264+k];
    const int u[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
    const float d8=sd[c*AS_XB_V264+k];
    float sum=0.0f;
#pragma unroll
    for(int i=0;i<2;++i){
        const int i8=i<<2;
        int sumi=ggml_cuda_dp4a(r.v0[i].x,u[i8+0],0);
        sumi=ggml_cuda_dp4a(r.v0[i].y,u[i8+2],sumi);
        sumi=ggml_cuda_dp4a(r.v1[i].x,u[i8+1],sumi);
        sumi=ggml_cuda_dp4a(r.v1[i].y,u[i8+3],sumi);
        sum += r.sc[i] * d8 * (float)sumi;
    }
    return sum;
}
template<int NC>
__device__ __forceinline__ void actstage_row_v264(const uint8_t* row,const int4* su,const float* sd,int t,float(&s)[NC]) {
#pragma unroll
    for(int c=0;c<NC;++c)s[c]=0.0f;
    for(int k=t;k<80;k+=32){
        const int kbx=k/2,iqs=4*(k%2);
        const Split<40>::W w=Split<40>::load(row,kbx,iqs);
#pragma unroll
        for(int c=0;c<NC;++c)s[c] += actstage_apply_v264(w,su,sd,c,k);
    }
    float s16[NC];
#pragma unroll
    for(int c=0;c<NC;++c)s16[c]=0.0f;
    for(int k=t+16;k<64;k+=32){
        const int kbx=k/2,iqs=4*(k%2);
        const Split<40>::W w=Split<40>::load(row,kbx,iqs);
#pragma unroll
        for(int c=0;c<NC;++c)s16[c] += actstage_apply_v264(w,su,sd,c,k);
    }
#pragma unroll
    for(int c=0;c<NC;++c){
        s[c]=__fadd_rn(s[c],s16[c]);
#pragma unroll
        for(int o=8;o>0;o>>=1)s[c] += __shfl_xor_sync(0xffffffffu,s[c],o);
    }
}
__global__ void __launch_bounds__(256) native_gu_actstage_v264_kernel(const unsigned long long* __restrict__ grp_ptr,
    const int32_t* __restrict__ grp_start,const int32_t* __restrict__ n_groups,const int32_t* __restrict__ ent_tok,
    const block_q8_1* __restrict__ xq,NativeExpertLayout L,float* __restrict__ gate,float* __restrict__ up) {
    const int ng=*n_groups;
    if(blockIdx.y>=ng)return; // uniform block return BEFORE any barrier
    // Host admission fixes grid.x=80,n_ff=640: all 16 rows in every block exist.
    __shared__ alignas(16) int4 su[AS_NC_V264*2*AS_XB_V264];
    __shared__ float sd[AS_NC_V264*AS_XB_V264];
    const int tid=threadIdx.x,t=tid&15,row=blockIdx.x*16+(tid>>4);
    const bool is_up=row>=640;const int r=is_up?row-640:row;
    float* dst=is_up?up:gate;
    const size_t woff=(is_up?L.up_off:0)+(size_t)r*L.gu_row;
    for(int g=blockIdx.y;g<ng;g+=gridDim.y){
        const uint8_t* wr=(const uint8_t*)grp_ptr[g]+woff;
        const float sc=L.tail_off?((const float*)(grp_ptr[g]+L.tail_off))[is_up?1:0]:1.0f;
        const int e0=grp_start[g],e1=grp_start[g+1];
        for(int e=e0;e<e1;e+=AS_NC_V264){
            const int n=min(AS_NC_V264,e1-e);
            __syncthreads(); // previous consumers release this CTA-local tile
            for(int c=0;c<n;++c)actstage_copy_v264(su,sd,c,xq+(size_t)ent_tok[e+c]*AS_XB_V264,tid);
            __syncthreads(); // all original raw codes + exact half d conversions visible
            auto run=[&](auto nc){
                constexpr int NC=decltype(nc)::value;float s[NC];actstage_row_v264<NC>(wr,su,sd,t,s);
#pragma unroll
                for(int c=0;c<NC;++c)if(t==c)dst[(size_t)(e+c)*640+r]=s[c]*sc;
            };
            if(n==1)run(std::integral_constant<int,1>{});
            else if(n==2)run(std::integral_constant<int,2>{});
            else if(n==3)run(std::integral_constant<int,3>{});
            else run(std::integral_constant<int,4>{});
        }
    }
}
