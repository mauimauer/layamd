#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <chrono>
__global__ void k_read(const uint4* p,size_t n,int never,unsigned* s){ unsigned a=0; size_t st=(size_t)gridDim.x*blockDim.x;
  for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=st) a^=p[i].x; if(never) s[threadIdx.x]=a; }
int main(){
  hipblasHandle_t h; hipblasCreate(&h); float one=1,zero=0;
  const int M=64,N=5248,K=1024; size_t wb=(size_t)N*K*2;
  void *A,*C,*flush; unsigned* s; hipMalloc(&A,M*K*2); hipMalloc(&C,M*N*2); hipMalloc(&flush,256<<20); hipMalloc(&s,1024);
  hipMemset(A,0,M*K*2); hipMemset(flush,1,256<<20);
  const int NW=16; void* W[NW]; for(int i=0;i<NW;i++){ hipMalloc(&W[i],wb); hipMemset(W[i],0,wb); }
  hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b);
  auto gemm=[&](void* w){ hipblasGemmEx(h,HIPBLAS_OP_N,HIPBLAS_OP_N,N,M,K,&one,w,HIP_R_16F,N,A,HIP_R_16F,K,&zero,C,HIP_R_16F,N,HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT); };
  auto doflush=[&]{ hipLaunchKernelGGL(k_read,dim3(1024),dim3(256),0,0,(const uint4*)flush,(size_t)(256<<20)/16,0,s); };
  auto timed=[&](auto f){ hipEventRecord(a); f(); hipEventRecord(b); hipEventSynchronize(b); float ms; hipEventElapsedTime(&ms,a,b); return ms*1000; };
  for(int i=0;i<3;i++) gemm(W[0]); hipDeviceSynchronize();
  double cold=0,warm=0,hot=0,pft=0;
  for(int i=0;i<NW;i++){ doflush(); hipDeviceSynchronize(); cold+=timed([&]{gemm(W[i]);}); }
  for(int i=0;i<NW;i++){ doflush(); hipDeviceSynchronize();
    pft+=timed([&]{ hipLaunchKernelGGL(k_read,dim3(1024),dim3(256),0,0,(const uint4*)W[i],wb/16,0,s); });
    warm+=timed([&]{gemm(W[i]);}); }
  for(int i=0;i<NW;i++){ gemm(W[i]); hipDeviceSynchronize(); hot+=timed([&]{gemm(W[i]);}); }
  printf("GEMM %dx%dx%d (W %.1f MB)\n cold (LLC flushed):     %6.1f us\n after read-prefetch:    %6.1f us  (prefetch itself %.1f us = %.0f GB/s)\n hot (just used):        %6.1f us\n",
         M,N,K,wb/1e6,cold/NW,warm/NW,pft/NW,wb/(pft/NW)/1e3,hot/NW);
}
