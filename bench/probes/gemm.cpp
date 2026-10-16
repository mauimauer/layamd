#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <vector>
#include <string>
#include <chrono>
#define HC(x) do{auto e=(x); if(e!=hipSuccess){printf("HIP err %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} }while(0)
#define BC(x) do{auto e=(x); if(e!=HIPBLAS_STATUS_SUCCESS){printf("BLAS err %d @%d\n",(int)e,__LINE__);exit(1);} }while(0)

__global__ void nullk(){}

struct Shape{int M,N,K; const char* name;};

int main(){
  hipblasHandle_t h; BC(hipblasCreate(&h));
  // ---- launch overhead ----
  {
    hipStream_t s; HC(hipStreamCreate(&s));
    for(int i=0;i<100;i++) hipLaunchKernelGGL(nullk,dim3(1),dim3(64),0,s);
    HC(hipStreamSynchronize(s));
    int N=20000; auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++) hipLaunchKernelGGL(nullk,dim3(1),dim3(64),0,s);
    HC(hipStreamSynchronize(s));
    auto t1=std::chrono::high_resolution_clock::now();
    double us=std::chrono::duration<double,std::micro>(t1-t0).count()/N;
    printf("Empty kernel launch (async, amortized): %.2f us\n", us);
  }
  // ---- memory bandwidth (read+write stream) ----
  {
    size_t bytes= 512ull<<20; // 512MB
    void *a,*b; HC(hipMalloc(&a,bytes)); HC(hipMalloc(&b,bytes));
    HC(hipMemset(a,1,bytes));
    for(int i=0;i<3;i++) HC(hipMemcpyDtoD(b,a,bytes));
    HC(hipDeviceSynchronize());
    int N=20; auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++) HC(hipMemcpyDtoD(b,a,bytes));
    HC(hipDeviceSynchronize());
    auto t1=std::chrono::high_resolution_clock::now();
    double sec=std::chrono::duration<double>(t1-t0).count()/N;
    printf("D2D bandwidth (r+w): %.1f GB/s\n", 2.0*bytes/sec/1e9);
    hipFree(a);hipFree(b);
  }
  printf("\n%-22s %6s %6s %6s  %9s %9s\n","op","M","N","K","time_us","TFLOPS");
  std::vector<Shape> shapes;
  for(int M : {128,256,512,1024,2048,4096}){
    shapes.push_back({M,3072,1024,"enc.Wqkv"});
    shapes.push_back({M,1024,1024,"enc.attn.Wo"});
    shapes.push_back({M,5248,1024,"enc.mlp.Wi"});
    shapes.push_back({M,1024,2624,"enc.mlp.Wo"});
    shapes.push_back({M,4096,1024,"head.linear1"});
    shapes.push_back({M,1024,4096,"head.linear2"});
  }
  int lastM=-1;
  for(auto&sh:shapes){
    if(sh.M!=lastM){printf("  --- seq/M = %d ---\n",sh.M);lastM=sh.M;}
    size_t szA=(size_t)sh.M*sh.K, szB=(size_t)sh.K*sh.N, szC=(size_t)sh.M*sh.N;
    __half *dA; __half *dB; __half *dC;
    HC(hipMalloc(&dA,szA*2));HC(hipMalloc(&dB,szB*2));HC(hipMalloc(&dC,szC*2));
    HC(hipMemset(dA,0,szA*2));HC(hipMemset(dB,0,szB*2));
    float alpha=1.f,beta=0.f;
    // C[M,N] = A[M,K]*B[K,N] ; column-major trick: compute Ct = Bt*At
    auto run=[&](){ BC(hipblasGemmEx(h,HIPBLAS_OP_N,HIPBLAS_OP_N,sh.N,sh.M,sh.K,&alpha,
        dB,HIP_R_16F,sh.N, dA,HIP_R_16F,sh.K, &beta, dC,HIP_R_16F,sh.N,
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT)); };
    for(int i=0;i<20;i++) run();
    HC(hipDeviceSynchronize());
    int iters = sh.M<=512?300:60;
    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<iters;i++) run();
    HC(hipDeviceSynchronize());
    auto t1=std::chrono::high_resolution_clock::now();
    double us=std::chrono::duration<double,std::micro>(t1-t0).count()/iters;
    double tflops=2.0*sh.M*sh.N*sh.K/(us*1e-6)/1e12;
    printf("%-22s %6d %6d %6d  %9.1f %9.2f\n",sh.name,sh.M,sh.N,sh.K,us,tflops);
    hipFree(dA);hipFree(dB);hipFree(dC);
  }
  hipblasDestroy(h);
  return 0;
}
