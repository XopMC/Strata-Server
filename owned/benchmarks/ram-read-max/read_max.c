#define _GNU_SOURCE
#include <errno.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <linux/mempolicy.h>
#include <sys/syscall.h>

/* Owned synthetic read-only bandwidth probe. Eight real 64-byte AVX512 loads
 * per iteration. Volatile asm + memory clobber prevents dead-load elimination;
 * no arithmetic/store is credited as read traffic. Units are decimal GB/s.
 * This measures delivered CPU load bandwidth, not DRAM controller counters.
 */
static void read_chunk(const unsigned char *p, size_t n) {
    size_t i=0;
    for (; i+512<=n; i+=512) {
        __asm__ volatile(
            "vmovdqu64 0(%0), %%zmm0\n\t"
            "vmovdqu64 64(%0), %%zmm1\n\t"
            "vmovdqu64 128(%0), %%zmm2\n\t"
            "vmovdqu64 192(%0), %%zmm3\n\t"
            "vmovdqu64 256(%0), %%zmm4\n\t"
            "vmovdqu64 320(%0), %%zmm5\n\t"
            "vmovdqu64 384(%0), %%zmm6\n\t"
            "vmovdqu64 448(%0), %%zmm7\n\t"
            : : "r"(p+i) : "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","memory");
    }
    for (;i<n;i+=64)
        __asm__ volatile("vmovdqu64 (%0), %%zmm0" : : "r"(p+i) : "zmm0","memory");
}
int main(void) {
    if (!__builtin_cpu_supports("avx512f")) { fprintf(stderr,"AVX512F required\n"); return 2; }
    const size_t bytes=(size_t)8*1024*1024*1024, pages=bytes/4096;
    unsigned char *buf=NULL;
    if (posix_memalign((void**)&buf,4096,bytes)) { perror("allocation"); return 2; }
    int n=omp_get_max_threads();
    if(n<1||n>504) { fprintf(stderr,"threads must be1..504\n"); free(buf); return 2; }
    double *starts=calloc(n,sizeof(double)),*ends=calloc(n,sizeof(double));
    int *cpus=calloc(n,sizeof(int));
    if(!starts||!ends||!cpus) return 2;
    int actual=0, admission_failed=0;
    #pragma omp parallel shared(actual,admission_failed)
    {
        int t=omp_get_thread_num(),nt=omp_get_num_threads();
        size_t lo=(pages*(size_t)t/(size_t)nt)*4096;
        size_t hi=(pages*(size_t)(t+1)/(size_t)nt)*4096;
        int socket=(sched_getcpu()%256)>=128; unsigned long mask=1UL<<socket;
        if(syscall(SYS_mbind,buf+lo,hi-lo,MPOL_BIND,&mask,64UL,0UL)) { perror("child-buffer-local-mbind"); _exit(3); }
        volatile uint64_t *v=(volatile uint64_t*)(buf+lo);
        for(size_t j=0;j<(hi-lo)/8;++j) v[j]=(uint64_t)(j+t+1);
        cpus[t]=sched_getcpu();
        #pragma omp barrier
        #pragma omp single
        {
            actual=nt;
            printf("{\"event\":\"ready\",\"bytes\":%zu,\"address\":%llu,\"threads\":%d,\"cpus\":[",bytes,(unsigned long long)(uintptr_t)buf,nt);
            for(int j=0;j<nt;++j) printf("%s%d",j?",":"",cpus[j]);
            puts("]}"); fflush(stdout);
            /* Root controller observes resident NUMA placement before release. */
            if(getchar()!='\n') admission_failed=1;
        }
        #pragma omp barrier
        if(!admission_failed) {
            for(int rep=-1;rep<7;++rep) {
                #pragma omp barrier
                starts[t]=omp_get_wtime();
                read_chunk(buf+lo,hi-lo);
                ends[t]=omp_get_wtime();
                #pragma omp barrier
                #pragma omp single
                {
                    double first=starts[0],last=ends[0];
                    for(int j=1;j<nt;++j) { if(starts[j]<first) first=starts[j]; if(ends[j]>last) last=ends[j]; }
                    printf("{\"event\":\"sample\",\"warmup\":%s,\"rep\":%d,\"bytes\":%zu,\"seconds\":%.9f,\"GB_s\":%.6f}\n",rep<0?"true":"false",rep,bytes,last-first,(double)bytes/(last-first)/1e9);
                    fflush(stdout);
                }
            }
        }
    }
    free(cpus);free(ends);free(starts);free(buf);
    return admission_failed||actual!=n?2:0;
}
