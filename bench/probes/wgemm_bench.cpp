#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include "../../src/wgemm.h"
#define CK(x) do{auto e=(x); if((int)e){printf("err %d @%d\n",(int)e,__LINE__);exit(1);} }while(0)
int main(){
  hipblasHandle_t hb; CK(hipblasCreate(&hb)); float one=1,zero=0;
  const int MAXM=1024, MAXN=5248, MAXK=4096;
  std::vector<__half> ha((size_t)MAXM*MAXK), hw((size_t)MAXN*MAXK);
  std::mt19937 rng(1); std::normal_distribution<float> nd(0,1);
  for(auto& v:ha) v=__float2half(nd(rng)); for(auto& v:hw) v=__float2half(nd(rng)*0.03f);
  __half *A,*Wt,*W,*C,*R,*G;
  CK(hipMalloc(&A,ha.size()*2)); CK(hipMalloc(&Wt,hw.size()*2)); CK(hipMalloc(&W,hw.size()*2));
  CK(hipMalloc(&C,(size_t)8*MAXM*MAXN*2)); CK(hipMalloc(&R,(size_t)MAXM*MAXN*2)); CK(hipMalloc(&G,(size_t)MAXM*MAXN*2));
  CK(hipMemcpy(A,ha.data(),ha.size()*2,hipMemcpyHostToDevice)); CK(hipMemcpy(Wt,hw.data(),hw.size()*2,hipMemcpyHostToDevice));
  hipEvent_t a,b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  auto T=[&](auto f){ for(int i=0;i<5;i++) f(); CK(hipDeviceSynchronize()); CK(hipEventRecord(a));
    for(int i=0;i<30;i++) f(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms,a,b)); return ms*1000/30; };
  // reference: TN hipBLAS on the same [N,K] weights
  auto ref=[&](int M,int N,int K,__half* out){ CK(hipblasGemmEx(hb,HIPBLAS_OP_T,HIPBLAS_OP_N,N,M,K,&one,Wt,HIP_R_16F,K,A,HIP_R_16F,K,&zero,out,HIP_R_16F,N,HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT)); };
  auto maxerr=[&](const __half* x,const __half* y,size_t n){ std::vector<__half> p(n),q(n);
    CK(hipMemcpy(p.data(),x,n*2,hipMemcpyDeviceToHost)); CK(hipMemcpy(q.data(),y,n*2,hipMemcpyDeviceToHost));
    double e=0,m=0; for(size_t i=0;i<n;i++){ e=fmax(e,fabs(__half2float(p[i])-__half2float(q[i]))); m=fmax(m,fabs(__half2float(q[i]))); } return e/m; };
  printf("%5s %5s %5s | %-14s %8s %7s | best split\n","M","N","K","kernel","us","TFLOPS");
  int shapes[][3]={{3072,1024,0},{1024,1024,0},{1024,2624,0},{4096,1024,0},{1024,4096,0}};
  for(int M:{64,128,256,512,1024}){
    for(auto& s:shapes){ int N=s[0],K=s[1];
      ref(M,N,K,R); CK(hipDeviceSynchronize());
      double best=1e30; int bc=-1,bs=1;
      for(int cfg=0;cfg<wgemm_ncfg();cfg++) for(int sp:{1,2,4,8}){
        if(!wgemm_supported(cfg,M,N,K,false,0) || K/64<sp) continue;
        launch_wgemm(cfg,A,Wt,C,M,N,K,sp,false,0,0); CK(hipDeviceSynchronize());
        if(sp==1){ double e=maxerr(C,R,(size_t)M*N); if(e>2e-3){ printf("  MISMATCH cfg%d rel err %.2e\n",cfg,e); } }
        double t=T([&]{ launch_wgemm(cfg,A,Wt,C,M,N,K,sp,false,0,0); });
        if(t<best){best=t;bc=cfg;bs=sp;}
      }
      double f=2.0*M*N*K;
      printf("%5d %5d %5d | wgemm cfg%-5d %8.1f %7.1f | %d\n",M,N,K,bc,best,f/best/1e6,bs);
    }
    // GeGLU fused vs (plain GEMM of 2I cols + separate geglu), I=2624
    { const int I=2624,K=1024;
      launch_wgemm(M>=128?4:1,A,Wt,G,M,2*I,K,1,true,I,0); CK(hipDeviceSynchronize());
      ref(M,2*I,K,R); CK(hipDeviceSynchronize());
      std::vector<__half> r((size_t)M*2*I), g((size_t)M*I);
      CK(hipMemcpy(r.data(),R,r.size()*2,hipMemcpyDeviceToHost)); CK(hipMemcpy(g.data(),G,g.size()*2,hipMemcpyDeviceToHost));
      double e=0,mx=0; for(int m=0;m<M;m++) for(int c=0;c<I;c++){
        float x=__half2float(r[(size_t)m*2*I+c]), y=__half2float(r[(size_t)m*2*I+I+c]);
        float want=0.5f*x*(1.f+erff(x*0.70710678f))*y; e=fmax(e,fabs(want-__half2float(g[(size_t)m*I+c]))); mx=fmax(mx,fabs(want)); }
      double best=1e30; int bc=0;
      for(int cfg=0;cfg<wgemm_ncfg();cfg++){ if(!wgemm_supported(cfg,M,2*I,K,true,I)) continue;
        double t=T([&]{ launch_wgemm(cfg,A,Wt,G,M,2*I,K,1,true,I,0); }); if(t<best){best=t;bc=cfg;} }
      printf("%5d %5d %5d | geglu cfg%-5d %8.1f %7.1f | rel err %.1e\n",M,2*I,K,bc,best,2.0*M*2*I*K/best/1e6,e/mx);
    }
  }
}
