#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <chrono>
#include <initializer_list>
#define BC(x) do{auto e=(x); if(e!=HIPBLAS_STATUS_SUCCESS){printf("BLAS %d @%d\n",(int)e,__LINE__);exit(1);} }while(0)
int main(){
  hipblasHandle_t h; BC(hipblasCreate(&h)); float one=1,zero=0;
  void *A,*W,*C; (void)hipMalloc(&A,8192*4096*2); (void)hipMalloc(&W,4096*4096*2); (void)hipMalloc(&C,8*8192*1024*4);
  (void)hipMemset(A,0,8192*4096*2); (void)hipMemset(W,0,4096*4096*2);
  auto T=[&](auto f){ for(int i=0;i<10;i++) f(); (void)hipDeviceSynchronize(); auto a=std::chrono::high_resolution_clock::now();
    for(int i=0;i<100;i++) f(); (void)hipDeviceSynchronize();
    return std::chrono::duration<double,std::micro>(std::chrono::high_resolution_clock::now()-a).count()/100; };
  printf("%5s %5s %5s | %8s | %8s %8s %8s | %8s %8s\n","M","N","K","plain","sk2_h","sk4_h","sk8_h","sk2_f32","sk4_f32");
  int shapes[][3]={{64,1024,2624},{128,1024,2624},{256,1024,2624},{128,1024,1024},{256,1024,1024},{128,3072,1024},{128,5248,1024}};
  for(auto& s:shapes){ int M=s[0],N=s[1],K=s[2];
    double p=T([&]{ BC(hipblasGemmEx(h,HIPBLAS_OP_N,HIPBLAS_OP_N,N,M,K,&one,W,HIP_R_16F,N,A,HIP_R_16F,K,&zero,C,HIP_R_16F,N,HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT)); });
    double r[5]; int i=0;
    for(int ty=0;ty<2;ty++) for(int sp: ty? std::initializer_list<int>{2,4} : std::initializer_list<int>{2,4,8}){
      int kc=K/sp; hipDataType ct= ty? HIP_R_32F: HIP_R_16F;
      r[i++]=T([&]{ BC(hipblasGemmStridedBatchedEx(h,HIPBLAS_OP_N,HIPBLAS_OP_N,N,M,kc,&one,W,HIP_R_16F,N,(long long)kc*N,A,HIP_R_16F,K,(long long)kc,&zero,C,ct,N,(long long)M*N,sp,HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT)); }); }
    printf("%5d %5d %5d | %8.1f | %8.1f %8.1f %8.1f | %8.1f %8.1f\n",M,N,K,p,r[0],r[1],r[2],r[3],r[4]);
  }
}
