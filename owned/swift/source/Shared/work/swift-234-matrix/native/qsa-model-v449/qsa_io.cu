#include "qsa_model.hpp"

namespace qsa_model_v449 {
namespace {
// Kernel-only P2P transport: no memcpy nodes inside a conditional/child graph.
__global__ void inputs_kernel(const float* query, const float* dead, const int32_t* steps,
        float* pq, float* pd, int32_t* ps, const float* pooled, float* mirror, int64_t rows,
        const Control* control, const uint64_t* seeded_epoch, uint32_t* owner_error, int n) {
    __shared__ int valid;
    if (threadIdx.x == 0) {
        bool ok = control->epoch && control->epoch == *seeded_epoch && control->queries == n;
        const int64_t position = steps[0];
        ok = ok && position >= 0 && position == control->position;
        for (int q = 0; q < n; ++q) {
            const int64_t kv = position + q + 1;
            ok = ok && steps[q*4] == kv-1 && steps[q*4+1] == kv &&
                 steps[q*4+2] == kv/4 && steps[q*4+3] == min(kv, int64_t(2051));
        }
        valid = int(ok);
        if (!ok) { *owner_error = 1; __threadfence_system(); }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n*512; i += blockDim.x) pq[i] = valid ? query[i] : 0.0f;
    for (int i = threadIdx.x; i < 128; i += blockDim.x) pd[i] = valid ? dead[i] : 0.0f;
    for (int i = threadIdx.x; i < n*4; i += blockDim.x) ps[i] = valid ? steps[i] : 0;
    // A commit can overwrite the moving spare even when this append writes no
    // pooled row. Copy the physical [floor(p/4), floor((p+n)/4)] intersection.
    // Previous unaccepted future rows remain masked and are never trusted.
    if (valid) {
        const int64_t first = int64_t(steps[0])/4;
        const int64_t last = (int64_t(steps[0])+n)/4;
        for (int64_t row = first; row <= last && row < rows; ++row)
            for (int d = threadIdx.x; d < 128; d += blockDim.x)
                mirror[row*128+d] = pooled[row*128+d];
    }
}
__global__ void packet_kernel(const uint32_t* source, uint32_t* destination, int words) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < words; i += gridDim.x*blockDim.x)
        destination[i] = source[i];
}
__global__ void ids_kernel(const int32_t* source, int32_t* destination,
        const int32_t* original_steps, int local, int remote) {
    const int q = blockIdx.x;
    if (q < remote) for (int i = threadIdx.x; i < original_steps[(q+local)*4+3]; i += blockDim.x)
        destination[(q+local)*2051+i] = source[q*2051+i];
}
__global__ void error_ids_kernel(int32_t* ids, const int32_t* steps, const uint32_t* error) {
    if (*error) for (int i = threadIdx.x; i < steps[blockIdx.x*4+3]; i += blockDim.x)
        ids[blockIdx.x*2051+i] = 0;
}
}
cudaError_t peer_inputs(const float* q,const float* d,const int32_t* st,float* pq,float* pd,int32_t* ps,
        const float* pool,float* mirror,int64_t rows,const Control* c,const uint64_t* epoch,
        uint32_t* error,int n,cudaStream_t stream) {
    inputs_kernel<<<1,256,0,stream>>>(q,d,st,pq,pd,ps,pool,mirror,rows,c,epoch,error,n);
    return cudaGetLastError();
}
cudaError_t return_packet(const uint32_t* src,uint32_t* dst,int queries,int words,cudaStream_t stream) {
    packet_kernel<<<8,256,0,stream>>>(src,dst,queries*words); return cudaGetLastError();
}
cudaError_t return_ids(const int32_t* src,int32_t* dst,const int32_t* steps,int local,int remote,cudaStream_t stream) {
    ids_kernel<<<remote,256,0,stream>>>(src,dst,steps,local,remote); return cudaGetLastError();
}
cudaError_t bounded_error_ids(int32_t* ids,const int32_t* steps,const uint32_t* error,int queries,cudaStream_t stream) {
    error_ids_kernel<<<queries,256,0,stream>>>(ids,steps,error); return cudaGetLastError();
}
}
