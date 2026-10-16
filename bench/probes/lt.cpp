#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
#include <chrono>
#include <vector>
#define CK(x) do{auto e=(x); if((int)e!=0){printf("err %d @%d\n",(int)e,__LINE__);exit(1);} }while(0)
int main(){
  hipblasLtHandle_t lt; CK(hipblasLtCreate(&lt)); hipblasHandle_t hb; CK(hipblasCreate(&hb));
  void *A,*W,*C,*ws; size_t wsz=64<<20; CK(hipMalloc(&A,8192*4096*2)); CK(hipMalloc(&W,4096*5248*2)); CK(hipMalloc(&C,8192*5248*2)); CK(hipMalloc(&ws,wsz));
  CK(hipMemset(A,0,8192*4096*2)); CK(hipMemset(W,0,4096*5248*2));
  float one=1,zero=0;
  auto T=[&](auto f){ for(int i=0;i<5;i++) f(); CK(hipDeviceSynchronize()); auto a=std::chrono::high_resolution_clock::now();
    for(int i=0;i<50;i++) f(); CK(hipDeviceSynchronize());
    return std::chrono::duration<double,std::micro>(std::chrono::high_resolution_clock::now()-a).count()/50; };
  printf("%5s %5s %5s | %9s | %9s %5s | %s\n","M","N","K","hipBLAS","LT best","#alg","TFLOPS blas->lt");
  int shapes[][3]={{128,3072,1024},{128,1024,2624},{256,5248,1024},{1024,3072,1024},{1024,1024,1024},{1024,5248,1024},{1024,1024,2624},{2048,5248,1024},{4096,5248,1024},{4096,1024,2624}};
  for(auto&s:shapes){ int M=s[0],N=s[1],K=s[2];
    double tb=T([&]{ CK(hipblasGemmEx(hb,HIPBLAS_OP_N,HIPBLAS_OP_N,N,M,K,&one,W,HIP_R_16F,N,A,HIP_R_16F,K,&zero,C,HIP_R_16F,N,HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT)); });
    hipblasLtMatmulDesc_t d; CK(hipblasLtMatmulDescCreate(&d,HIPBLAS_COMPUTE_32F,HIP_R_32F));
    hipblasOperation_t op=HIPBLAS_OP_N;
    CK(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSA,&op,sizeof op));
    CK(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSB,&op,sizeof op));
    hipblasLtMatrixLayout_t la,lb,lc;
    CK(hipblasLtMatrixLayoutCreate(&la,HIP_R_16F,N,K,N)); CK(hipblasLtMatrixLayoutCreate(&lb,HIP_R_16F,K,M,K)); CK(hipblasLtMatrixLayoutCreate(&lc,HIP_R_16F,N,M,N));
    hipblasLtMatmulPreference_t pr; CK(hipblasLtMatmulPreferenceCreate(&pr));
    CK(hipblasLtMatmulPreferenceSetAttribute(pr,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof wsz));
    hipblasLtMatmulHeuristicResult_t hr[32]; int n=0;
    CK(hipblasLtMatmulAlgoGetHeuristic(lt,d,la,lb,lc,lc,pr,32,hr,&n));
    double best=1e30;
    for(int i=0;i<n;i++){
      auto r=hipblasLtMatmul(lt,d,&one,W,la,A,lb,&zero,C,lc,C,lc,&hr[i].algo,ws,wsz,0);
      if(r!=0) continue;
      double t=T([&]{ hipblasLtMatmul(lt,d,&one,W,la,A,lb,&zero,C,lc,C,lc,&hr[i].algo,ws,wsz,0); });
      if(t<best) best=t;
    }
    double f=2.0*M*N*K;
    printf("%5d %5d %5d | %9.1f | %9.1f %5d | %5.1f -> %5.1f\n",M,N,K,tb,best,n,f/tb/1e6,f/best/1e6);
  }
}
