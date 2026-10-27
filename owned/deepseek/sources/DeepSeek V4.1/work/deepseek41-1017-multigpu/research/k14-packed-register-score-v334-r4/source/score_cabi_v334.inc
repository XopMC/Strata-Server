extern "C" int ds334_score_info(uint64_t*out,int words){
    using namespace strata::ds41::kernels;
    if(!out||words!=16)return -1;
    try{for(int i=0;i<16;i++)out[i]=score_stats334[i].load(std::memory_order_relaxed);
        out[0]=334;out[1]=score_flag334();out[10]=sizeof(ScoreTile334);out[11]=16;out[12]=128;out[13]=4;out[14]=512;return 0;}catch(...){return -2;}
}
extern "C" int ds334_score_resources(uint64_t*out,int words){
    using namespace strata::ds41::kernels;
    if(!out||words!=12)return -1;
    auto e=cudaFuncSetAttribute(batched_scores334,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(ScoreTile334));if(e!=cudaSuccess)return int(e);
    cudaFuncAttributes a{};e=cudaFuncGetAttributes(&a,batched_scores334);if(e!=cudaSuccess)return int(e);
    int active=0;e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,batched_scores334,512,sizeof(ScoreTile334));if(e!=cudaSuccess)return int(e);
    int device=-1;cudaDeviceProp prop{};e=cudaGetDevice(&device);if(e!=cudaSuccess)return int(e);e=cudaGetDeviceProperties(&prop,device);if(e!=cudaSuccess)return int(e);
    const uint64_t v[12]={334,sizeof(ScoreTile334),uint64_t(a.numRegs),uint64_t(active),uint64_t(a.maxThreadsPerBlock),16,128,4,uint64_t(a.localSizeBytes),512,uint64_t(prop.reservedSharedMemPerBlock),uint64_t(prop.sharedMemPerMultiprocessor)};
    std::memcpy(out,v,sizeof(v));return 0;
}
extern "C" int ds334_raw_scores(const void*q,const void*k,const void*w,int m,int pos,int ratio,void*out,int64_t stride,uint64_t stream_value){
    using namespace strata::ds41::kernels;
    try{
        if(!score_flag334()||!q||!k||!w||!out||m<1||m>256||pos<0||ratio!=2||stride<(int64_t(pos)+m)/2)return -1;
        if((uintptr_t(q)|uintptr_t(k)|uintptr_t(w)|uintptr_t(out))%2)return -1;
        const int64_t n=(int64_t(pos)+m)/2;if(!n)return 0;if(n>1048576||stride>1048576)return -1;
        auto range=[](const void*p,uint64_t bytes){auto a=reinterpret_cast<uintptr_t>(p);return a&&bytes&&bytes<=uint64_t(PTRDIFF_MAX)&&bytes<=UINTPTR_MAX-a;};
        const uint64_t qb=uint64_t(m)*32*128*2,kb=uint64_t(n)*128*2,wb=uint64_t(m)*32*2,ob=uint64_t(m)*uint64_t(stride)*2;
        if(!range(q,qb)||!range(k,kb)||!range(w,wb)||!range(out,ob))return -1;
        auto overlap=[&](const void*p,uint64_t bytes){auto a=reinterpret_cast<uintptr_t>(out),b=reinterpret_cast<uintptr_t>(p);return a<b+bytes&&b<a+ob;};
        if(overlap(q,qb)||overlap(k,kb)||overlap(w,wb))return -1;
        auto stream=reinterpret_cast<cudaStream_t>(stream_value);
        auto e=cudaFuncSetAttribute(batched_scores334,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(ScoreTile334));if(e!=cudaSuccess)return int(e);
        dim3 grid((n+127)/128,(m+15)/16);
        batched_scores334<<<grid,512,sizeof(ScoreTile334),stream>>>((const bf16*)q,(const bf16*)k,(const bf16*)w,0,m,pos,ratio,(bf16*)out,stride);
        record_launch334(m,uint64_t(grid.x)*grid.y,n);score_stats334[15].fetch_add(1,std::memory_order_relaxed);
        return int(cudaGetLastError());
    }catch(...){return -2;}
}
