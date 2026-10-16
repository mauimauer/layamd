// Timing skeleton for Laya (ModernBERT-large encoder + 2-layer decision head) on gfx1151.
// Real shapes, real kernels, dummy weights. Measures achievable latency + bottleneck split.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#define HC(x) do{auto e=(x); if(e!=hipSuccess){printf("HIP err %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} }while(0)
#define BC(x) do{auto e=(x); if(e!=HIPBLAS_STATUS_SUCCESS){printf("BLAS err %d @%d\n",(int)e,__LINE__);exit(1);} }while(0)

static const int D=1024, H=16, HD=64, I=2624, NL=28, HEADL=2, FF=4096, LOCAL=128;

// ---------- kernels ----------
__global__ void k_layernorm(const __half* __restrict__ x, const __half* __restrict__ w,
                            const __half* __restrict__ b, __half* __restrict__ y, int D_, float eps){
  int row=blockIdx.x; const __half* xr=x+(size_t)row*D_; __half* yr=y+(size_t)row*D_;
  __shared__ float ssum[256]; __shared__ float ssq[256];
  float s=0,q=0;
  for(int i=threadIdx.x;i<D_;i+=blockDim.x){ float v=__half2float(xr[i]); s+=v; q+=v*v; }
  ssum[threadIdx.x]=s; ssq[threadIdx.x]=q; __syncthreads();
  for(int o=blockDim.x/2;o>0;o>>=1){ if(threadIdx.x<o){ssum[threadIdx.x]+=ssum[threadIdx.x+o];ssq[threadIdx.x]+=ssq[threadIdx.x+o];} __syncthreads(); }
  float mean=ssum[0]/D_; float var=ssq[0]/D_-mean*mean; float inv=rsqrtf(var+eps);
  for(int i=threadIdx.x;i<D_;i+=blockDim.x){
    float v=(__half2float(xr[i])-mean)*inv*__half2float(w[i]);
    if(b) v+=__half2float(b[i]);
    yr[i]=__float2half(v);
  }
}
__global__ void k_rope(__half* __restrict__ qkv, const float* __restrict__ cs, int S,int H_,int HD_,int stride3D){
  int t=blockIdx.x, h=blockIdx.y, i=threadIdx.x; if(i>=HD_/2) return;
  const float* c=cs+(size_t)t*HD_; float cosv=c[2*i], sinv=c[2*i+1];
  for(int part=0;part<2;part++){ // q then k
    __half* p=qkv+(size_t)t*stride3D + part*(H_*HD_) + h*HD_;
    float a=__half2float(p[2*i]), b=__half2float(p[2*i+1]);
    p[2*i]=__float2half(a*cosv-b*sinv); p[2*i+1]=__float2half(a*sinv+b*cosv);
  }
}
__global__ void k_softmax_mask(__half* __restrict__ s,int S,int local_,int is_global){
  int row=blockIdx.x, h=blockIdx.y; __half* r=s+((size_t)h*S+row)*S;
  int lo=0,hi=S; if(!is_global){ lo=max(0,row-local_/2); hi=min(S,row+local_/2+1); }
  __shared__ float sm[256];
  float m=-1e30f;
  for(int i=threadIdx.x;i<S;i+=blockDim.x) if(i>=lo&&i<hi) m=fmaxf(m,__half2float(r[i]));
  sm[threadIdx.x]=m; __syncthreads();
  for(int o=blockDim.x/2;o>0;o>>=1){ if(threadIdx.x<o) sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+o]); __syncthreads(); }
  m=sm[0]; float acc=0;
  for(int i=threadIdx.x;i<S;i+=blockDim.x){ float v=(i>=lo&&i<hi)?__expf(__half2float(r[i])-m):0.f; r[i]=__float2half(v); acc+=v; }
  sm[threadIdx.x]=acc; __syncthreads();
  for(int o=blockDim.x/2;o>0;o>>=1){ if(threadIdx.x<o) sm[threadIdx.x]+=sm[threadIdx.x+o]; __syncthreads(); }
  float inv=1.f/fmaxf(sm[0],1e-9f);
  for(int i=threadIdx.x;i<S;i+=blockDim.x) r[i]=__float2half(__half2float(r[i])*inv);
}
__global__ void k_geglu(const __half* __restrict__ in, __half* __restrict__ out,int S,int I_){
  size_t n=(size_t)S*I_; for(size_t idx=blockIdx.x*(size_t)blockDim.x+threadIdx.x; idx<n; idx+=(size_t)gridDim.x*blockDim.x){
    int row=idx/I_, col=idx%I_;
    float a=__half2float(in[(size_t)row*2*I_+col]);       // gate half
    float b=__half2float(in[(size_t)row*2*I_+I_+col]);    // up half
    float g=0.5f*a*(1.f+tanhf(0.7978845608f*(a+0.044715f*a*a*a)));
    out[idx]=__float2half(g*b);
  }
}
__global__ void k_resadd(__half* __restrict__ x,const __half* __restrict__ y,size_t n){
  for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x)
    x[i]=__float2half(__half2float(x[i])+__half2float(y[i]));
}
__global__ void k_bias_relu(__half* __restrict__ x,const __half* __restrict__ b,int S,int N_){
  size_t n=(size_t)S*N_; for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){
    float v=__half2float(x[i])+__half2float(b[i%N_]); x[i]=__float2half(v>0?v:0.f);
  }
}
__global__ void k_transpose_heads(const __half* __restrict__ src,__half* __restrict__ dst,int S,int H_,int HD_,int off,int stride3D){
  // [S, 3D] -> [H, S, HD]
  int t=blockIdx.x,h=blockIdx.y;
  const __half* s=src+(size_t)t*stride3D+off+h*HD_;
  __half* d=dst+((size_t)h*S+t)*HD_;
  for(int i=threadIdx.x;i<HD_;i+=blockDim.x) d[i]=s[i];
}
__global__ void k_untranspose(const __half* __restrict__ src,__half* __restrict__ dst,int S,int H_,int HD_){
  int t=blockIdx.x,h=blockIdx.y;
  const __half* s=src+((size_t)h*S+t)*HD_; __half* d=dst+(size_t)t*(H_*HD_)+h*HD_;
  for(int i=threadIdx.x;i<HD_;i+=blockDim.x) d[i]=s[i];
}

struct Timer{
  hipEvent_t a,b; std::map<std::string,float> acc; bool on=false;
  Timer(){HC(hipEventCreate(&a));HC(hipEventCreate(&b));}
  void start(){ if(!on) return; HC(hipEventRecord(a));}
  void stop(const std::string& n){ if(!on) return; HC(hipEventRecord(b));HC(hipEventSynchronize(b));float ms;HC(hipEventElapsedTime(&ms,a,b));acc[n]+=ms;}
};

int main(int argc,char**argv){
  int S = argc>1?atoi(argv[1]):512;
  int BATCH = argc>2?atoi(argv[2]):1;
  int M = S*BATCH;
  bool detail = argc>3;
  hipblasHandle_t h; BC(hipblasCreate(&h));
  float alpha=1.f,beta=0.f;
  auto gemm=[&](const __half*A,const __half*B,__half*C,int m,int n,int k){
    BC(hipblasGemmEx(h,HIPBLAS_OP_N,HIPBLAS_OP_N,n,m,k,&alpha,B,HIP_R_16F,n,A,HIP_R_16F,k,&beta,C,HIP_R_16F,n,
        HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT));
  };
  auto alloc=[&](size_t n){ __half*p; HC(hipMalloc(&p,n*2)); HC(hipMemset(p,0,n*2)); return p; };
  // weights
  __half *Wqkv=alloc((size_t)3*D*D), *Wo=alloc((size_t)D*D), *Wi=alloc((size_t)2*I*D), *Wmo=alloc((size_t)D*I);
  __half *lnw=alloc(D), *lnb=alloc(D);
  __half *hin=alloc((size_t)3*D*D), *hout=alloc((size_t)D*D), *hl1=alloc((size_t)FF*D), *hl2=alloc((size_t)D*FF), *bff=alloc(FF);
  // activations
  __half *x=alloc((size_t)M*D), *xn=alloc((size_t)M*D), *qkv=alloc((size_t)M*3*D);
  __half *qt=alloc((size_t)M*D), *kt=alloc((size_t)M*D), *vt=alloc((size_t)M*D), *ot=alloc((size_t)M*D);
  __half *scores=alloc((size_t)H*S*S*BATCH), *ao=alloc((size_t)M*D);
  __half *mi=alloc((size_t)M*2*I), *mg=alloc((size_t)M*I), *mo=alloc((size_t)M*D);
  __half *ff=alloc((size_t)M*FF);
  float* cs; HC(hipMalloc(&cs,(size_t)S*HD*sizeof(float))); HC(hipMemset(cs,0,(size_t)S*HD*sizeof(float)));

  Timer T;
  auto attention=[&](int is_global){
    T.start();
    hipLaunchKernelGGL(k_transpose_heads,dim3(M),dim3(64),0,0,qkv,qt,M,H,HD,0,3*D);
    hipLaunchKernelGGL(k_transpose_heads,dim3(M),dim3(64),0,0,qkv,kt,M,H,HD,D,3*D);
    hipLaunchKernelGGL(k_transpose_heads,dim3(M),dim3(64),0,0,qkv,vt,M,H,HD,2*D,3*D);
    T.stop("attn:reshape");
    T.start();
    float sc=1.f/sqrtf((float)HD);
    // scores[h] = q[h] @ k[h]^T  : [S,HD]x[HD,S]
    BC(hipblasGemmStridedBatchedEx(h,HIPBLAS_OP_T,HIPBLAS_OP_N,S,S,HD,&sc,
       kt,HIP_R_16F,HD,(long long)S*HD, qt,HIP_R_16F,HD,(long long)S*HD,&beta,
       scores,HIP_R_16F,S,(long long)S*S,H*BATCH,HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT));
    T.stop("attn:qk");
    T.start();
    hipLaunchKernelGGL(k_softmax_mask,dim3(S,H*BATCH),dim3(256),0,0,scores,S,LOCAL,is_global);
    T.stop("attn:softmax");
    T.start();
    BC(hipblasGemmStridedBatchedEx(h,HIPBLAS_OP_N,HIPBLAS_OP_N,HD,S,S,&alpha,
       vt,HIP_R_16F,HD,(long long)S*HD, scores,HIP_R_16F,S,(long long)S*S,&beta,
       ot,HIP_R_16F,HD,(long long)S*HD,H*BATCH,HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT));
    T.stop("attn:av");
    T.start();
    hipLaunchKernelGGL(k_untranspose,dim3(M),dim3(64),0,0,ot,ao,M,H,HD);
    T.stop("attn:reshape");
  };

  auto forward=[&](){
    for(int L=0;L<NL;L++){
      if(L>0){ T.start(); hipLaunchKernelGGL(k_layernorm,dim3(M),dim3(256),0,0,x,lnw,(const __half*)nullptr,xn,D,1e-5f); T.stop("enc:layernorm"); }
      else HC(hipMemcpyDtoD(xn,x,(size_t)M*D*2));
      T.start(); gemm(xn,Wqkv,qkv,M,3*D,D); T.stop("enc:gemm_qkv");
      T.start(); hipLaunchKernelGGL(k_rope,dim3(M,H),dim3(HD/2),0,0,qkv,cs,M,H,HD,3*D); T.stop("enc:rope");
      attention(L%3==0);
      T.start(); gemm(ao,Wo,mo,M,D,D); T.stop("enc:gemm_attn_out");
      T.start(); hipLaunchKernelGGL(k_resadd,dim3(512),dim3(256),0,0,x,mo,(size_t)M*D); T.stop("enc:residual");
      T.start(); hipLaunchKernelGGL(k_layernorm,dim3(M),dim3(256),0,0,x,lnw,(const __half*)nullptr,xn,D,1e-5f); T.stop("enc:layernorm");
      T.start(); gemm(xn,Wi,mi,M,2*I,D); T.stop("enc:gemm_mlp_in");
      T.start(); hipLaunchKernelGGL(k_geglu,dim3(512),dim3(256),0,0,mi,mg,M,I); T.stop("enc:geglu");
      T.start(); gemm(mg,Wmo,mo,M,D,I); T.stop("enc:gemm_mlp_out");
      T.start(); hipLaunchKernelGGL(k_resadd,dim3(512),dim3(256),0,0,x,mo,(size_t)M*D); T.stop("enc:residual");
    }
    T.start(); hipLaunchKernelGGL(k_layernorm,dim3(M),dim3(256),0,0,x,lnw,(const __half*)nullptr,xn,D,1e-5f); T.stop("enc:layernorm");
    // ---- decision head: 2 x stock pytorch TransformerEncoderLayer (pre-norm, bias, ReLU) ----
    for(int L=0;L<HEADL;L++){
      T.start(); hipLaunchKernelGGL(k_layernorm,dim3(M),dim3(256),0,0,x,lnw,lnb,xn,D,1e-5f); T.stop("head:layernorm");
      T.start(); gemm(xn,hin,qkv,M,3*D,D); T.stop("head:gemm");
      attention(1);
      T.start(); gemm(ao,hout,mo,M,D,D); T.stop("head:gemm");
      T.start(); hipLaunchKernelGGL(k_resadd,dim3(512),dim3(256),0,0,x,mo,(size_t)M*D); T.stop("head:residual");
      T.start(); hipLaunchKernelGGL(k_layernorm,dim3(M),dim3(256),0,0,x,lnw,lnb,xn,D,1e-5f); T.stop("head:layernorm");
      T.start(); gemm(xn,hl1,ff,M,FF,D); T.stop("head:gemm");
      T.start(); hipLaunchKernelGGL(k_bias_relu,dim3(512),dim3(256),0,0,ff,bff,M,FF); T.stop("head:act");
      T.start(); gemm(ff,hl2,mo,M,D,FF); T.stop("head:gemm");
      T.start(); hipLaunchKernelGGL(k_resadd,dim3(512),dim3(256),0,0,x,mo,(size_t)M*D); T.stop("head:residual");
    }
  };

  T.on=false;
  for(int i=0;i<5;i++) forward();       // warmup
  HC(hipDeviceSynchronize());
  int iters=20;
  auto t0=std::chrono::high_resolution_clock::now();
  for(int i=0;i<iters;i++) forward();
  HC(hipDeviceSynchronize());
  auto t1=std::chrono::high_resolution_clock::now();
  double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/iters;
  // second pass: per-stage breakdown (serialized, inflates totals - use for ratios only)
  T.on=true; T.acc.clear();
  int biters=5; for(int i=0;i<biters;i++) forward();
  HC(hipDeviceSynchronize());
  iters=biters;

  printf("\n=== Laya fwd: seq=%d batch=%d (M=%d) ===\n",S,BATCH,M);
  double tot=0; for(auto&kv:T.acc) tot+=kv.second/iters;
  std::vector<std::pair<double,std::string>> v;
  for(auto&kv:T.acc) v.push_back({kv.second/iters,kv.first});
  std::sort(v.rbegin(),v.rend());
  printf("%-22s %9s %7s\n","stage","ms","%");
  for(auto&p:v) printf("%-22s %9.3f %6.1f%%\n",p.second.c_str(),p.first,100*p.first/tot);
  printf("%-22s %9.3f\n","SUM(kernels)",tot);
  printf("%-22s %9.3f   <-- wall clock end-to-end\n","TOTAL",ms);
  double q_s = 1000.0/ms*BATCH;
  printf("throughput: %.1f questions/sec\n", q_s);
  return 0;
}
