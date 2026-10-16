// Laya HIP kernels for gfx1151 (RDNA3.5, wave32).
// Rows are tokens of a padded batch: row = b*S + position.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include "kernels.h"

__device__ __forceinline__ float gelu_erf(float x){ return 0.5f*x*(1.f+erff(x*0.70710678118654752f)); }

// Block-wide sum/max for any blockDim that is a multiple of 32. `sh` needs blockDim/32 floats.
template<bool MAX>
__device__ __forceinline__ float block_red(float v, float* sh){
  #pragma unroll
  for(int o=16;o;o>>=1){ float t=__shfl_xor(v,o,32); v = MAX ? fmaxf(v,t) : v+t; }
  const int w=threadIdx.x>>5, nw=blockDim.x>>5;
  __syncthreads();                       // previous readers of sh are done
  if((threadIdx.x&31)==0) sh[w]=v;
  __syncthreads();
  float r=sh[0];
  for(int k=1;k<nw;k++) r = MAX ? fmaxf(r,sh[k]) : r+sh[k];
  return r;
}

// ---- LayerNorm, 128 threads/row, D % 256 == 0 && D <= 1024 (holds for 768 and 1024). ----
// Tin/Tout are float (the fp32 residual stream) or __half. src row = ids ? x[ids[row]]
// (embedding lookup) : x[row]. Optional bias, and optional per-sequence additive row
// addtab[addidx[rowseq[row]]] (the decision model's type embedding). Each thread keeps its slice in
// registers, so in-place (y == x) is safe.
__device__ __forceinline__ float2 ld2(const float* p,int i){ return ((const float2*)p)[i]; }
__device__ __forceinline__ float2 ld2(const __half* p,int i){ return __half22float2(((const __half2*)p)[i]); }
__device__ __forceinline__ void st2(float* p,int i,float2 v){ ((float2*)p)[i]=v; }
__device__ __forceinline__ void st2(__half* p,int i,float2 v){ ((__half2*)p)[i]=__float22half2_rn(v); }

template<typename Tin,typename Tout>
__global__ __launch_bounds__(128) void k_ln(const Tin* __restrict__ x, const int* __restrict__ ids,
    const __half* __restrict__ w, const __half* __restrict__ b, Tout* y, int D, float eps,
    const __half* __restrict__ addtab, const int* __restrict__ addidx, const int* __restrict__ rowseq){
  __shared__ float sh[4];
  const int row=blockIdx.x, n2=D/2;
  const size_t src = ids ? (size_t)ids[row] : (size_t)row;
  const Tin* xr=x+src*D;
  float2 v[4]; float s=0.f;
  #pragma unroll
  for(int k=0;k<4;k++){ int i=threadIdx.x+k*128; if(i<n2){ v[k]=ld2(xr,i); s+=v[k].x+v[k].y; } }
  const float mean=block_red<false>(s,sh)/D;
  float q=0.f;
  #pragma unroll
  for(int k=0;k<4;k++){ int i=threadIdx.x+k*128; if(i<n2){ float a=v[k].x-mean, c=v[k].y-mean; q+=a*a+c*c; } }
  const float inv=rsqrtf(block_red<false>(q,sh)/D+eps);
  const __half* a2 = addtab ? addtab+(size_t)addidx[rowseq[row]]*D : nullptr;
  Tout* yr=y+(size_t)row*D;
  #pragma unroll
  for(int k=0;k<4;k++){ int i=threadIdx.x+k*128; if(i<n2){
    float2 ww=ld2(w,i);
    float2 o=make_float2((v[k].x-mean)*inv*ww.x,(v[k].y-mean)*inv*ww.y);
    if(b){ float2 bb=ld2(b,i); o.x+=bb.x; o.y+=bb.y; }
    if(a2){ float2 aa=ld2(a2,i); o.x+=aa.x; o.y+=aa.y; }
    st2(yr,i,o);
  } }
}

// ---- RoPE (HF rotate_half) applied in place to the q and k thirds of qkv[rows][3D]. ----
// One block per row, one thread per (head, pair). V is untouched; attention reads Q/K/V
// straight out of qkv, so there are no split/transpose copies.
__global__ void k_rope_inplace(__half* __restrict__ qkv, const float* __restrict__ cosv,
                               const float* __restrict__ sinv, const int* __restrict__ pos_of, int H, int HD){
  const int row=blockIdx.x, H2=HD/2, h=threadIdx.x/H2, d=threadIdx.x%H2;
  if(h>=H) return;
  const int pos=pos_of[row], D=H*HD;
  const float c=cosv[pos*H2+d], sn=sinv[pos*H2+d];
  __half* q=qkv+(size_t)row*3*D+h*HD;
  #pragma unroll
  for(int part=0;part<2;part++){
    __half* p=q+part*D;
    float a=__half2float(p[d]), bb=__half2float(p[d+H2]);
    p[d]=__float2half(a*c-bb*sn); p[d+H2]=__float2half(bb*c+a*sn);
  }
}

// ---- split-K / residual reduction: dst = (accumulate ? dst : 0) + sum_s part[s] (+ bias) ----
// Small-M GEMMs produce too few output tiles to fill 40 CUs, so they run as `splits` K-slices
// (one strided-batched GEMM with fp16 partials -- fp32 GEMM output is a 2-4x slower path in
// hipBLAS on gfx1151) summed here in fp32. With TD=float this is also how every residual
// update lands in the fp32 residual stream (splits=1: part is the plain GEMM output).
template<typename TD>
__global__ void k_splitk_reduce(TD* __restrict__ dst, const __half* __restrict__ part, int splits,
                                size_t n, int N, const __half* __restrict__ bias, int accumulate){
  const size_t n2=n/2;
  for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n2;i+=(size_t)gridDim.x*blockDim.x){
    float2 acc = accumulate ? ld2(dst+2*i,0) : make_float2(0.f,0.f);
    for(int s=0;s<splits;s++){ float2 p=ld2(part+(size_t)s*n+2*i,0); acc.x+=p.x; acc.y+=p.y; }
    if(bias){ float2 bb=ld2(bias+(2*i)%N,0); acc.x+=bb.x; acc.y+=bb.y; }
    st2(dst+2*i,0,acc);
  }
}

// ---- fused residual update + LayerNorm, one block (128 threads) per row, D%256==0, D<=1024 ----
// x[row] += sum_s part[s][row] (+ bias); then y[row] = LN(x[row]) * w (+ b) (+ addtab[addidx[seq]]).
// Replaces "reduce kernel, then LayerNorm kernel": saves one pass over the fp32 residual and a
// launch per LayerNorm. y may alias x (final norm is written back into the fp32 stream).
template<typename Tout>
__global__ __launch_bounds__(128) void k_res_ln(float* __restrict__ x, const __half* __restrict__ part, int splits,
    size_t n, const __half* __restrict__ bias, const __half* __restrict__ w, const __half* __restrict__ lb,
    Tout* y, int D, float eps, const __half* __restrict__ addtab, const int* __restrict__ addidx,
    const int* __restrict__ rowseq){
  __shared__ float sh[4];
  const int row=blockIdx.x, n2=D/2;
  float* xr=x+(size_t)row*D;
  float2 v[4]; float s=0.f;
  #pragma unroll
  for(int k=0;k<4;k++){ int i=threadIdx.x+k*128; if(i<n2){
    float2 a=ld2(xr,i);
    for(int p=0;p<splits;p++){ float2 q=ld2(part+(size_t)p*n+(size_t)row*D,i); a.x+=q.x; a.y+=q.y; }
    if(bias){ float2 bb=ld2(bias,i); a.x+=bb.x; a.y+=bb.y; }
    v[k]=a; s+=a.x+a.y;
  } }
  const float mean=block_red<false>(s,sh)/D;
  float q=0.f;
  #pragma unroll
  for(int k=0;k<4;k++){ int i=threadIdx.x+k*128; if(i<n2){ float a=v[k].x-mean, c=v[k].y-mean; q+=a*a+c*c; } }
  const float inv=rsqrtf(block_red<false>(q,sh)/D+eps);
  const __half* a2 = addtab ? addtab+(size_t)addidx[rowseq[row]]*D : nullptr;
  Tout* yr=y+(size_t)row*D;
  #pragma unroll
  for(int k=0;k<4;k++){ int i=threadIdx.x+k*128; if(i<n2){
    if((void*)yr!=(void*)xr) st2(xr,i,v[k]);         // residual stream keeps the un-normalized sum
    float2 ww=ld2(w,i);
    float2 o=make_float2((v[k].x-mean)*inv*ww.x,(v[k].y-mean)*inv*ww.y);
    if(lb){ float2 bb=ld2(lb,i); o.x+=bb.x; o.y+=bb.y; }
    if(a2){ float2 aa=ld2(a2,i); o.x+=aa.x; o.y+=aa.y; }
    st2(yr,i,o);
  } }
}

// ---- fused flash attention on RDNA3 WMMA (v_wmma_f32_16x16x16_f16, wave32). ----
// Sequences are packed back to back (each padded to a multiple of 16 rows); a tile map gives
// each 16-query tile its sequence and local offset. One wave owns one tile of one head. For each 16-key block:
//   S = Q K^T via HD/16 WMMAs (fp32 accumulate) -> scale, mask, online softmax in fp32 registers
//   O += P V via HD/16 WMMAs.
// The S x S score matrix never touches memory, sliding-window layers only visit key blocks
// inside their band, and scores stay fp32 end to end. Q/K/V are read straight from
// qkv[rows][3D]; output goes to out[rows][D] (heads merged).
// WMMA wave32 fragment layouts (gfx11): A: lane l holds row l%16 (16 K-values), B: lane l holds
// column l%16 (16 K-values), both replicated in lanes 16-31; C/D: element i of lane l is
// D[2i + l/16][l%16].
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef _Float16 h8v __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));

template<int HD>
__global__ __launch_bounds__(128) void k_flash_wmma(const __half* __restrict__ qkv, __half* __restrict__ out,
    int ntiles, int H, int window, const int* __restrict__ vlen, const int* __restrict__ soff,
    const int* __restrict__ tseq, const int* __restrict__ tq0, float scale){
  constexpr int NC=HD/16;
  __shared__ _Float16 Vs[4][16][HD];
  __shared__ _Float16 Ps[4][16][16];
  const int wave=threadIdx.x>>5, lane=threadIdx.x&31, r16=lane&15, hl=lane>>4;
  const int h=blockIdx.y, D=H*HD, ti=blockIdx.x*4+wave;
  if(ti>=ntiles) return;                 // no block-wide barriers below: LDS slices are per wave
  const int b=tseq[ti];
  if(b<0) return;                        // padding tile at the end of the pack
  const int q0=tq0[ti], valid=vlen[b];
  const _Float16* base=(const _Float16*)qkv+(size_t)soff[b]*3*D;
  h16 qa[NC];
  #pragma unroll
  for(int c=0;c<NC;c++) qa[c]=*(const h16*)(base+(size_t)(q0+r16)*3*D+h*HD+c*16);
  int lo=0, hi=valid;
  if(window>0){ int hf=window/2; lo=max(0,q0-hf); hi=min(valid,q0+15+hf+1); }
  lo&=~15;
  f8 o[NC];
  #pragma unroll
  for(int c=0;c<NC;c++) o[c]=(f8){0,0,0,0,0,0,0,0};
  float m[8], l[8];
  #pragma unroll
  for(int i=0;i<8;i++){ m[i]=-INFINITY; l[i]=0.f; }
  for(int k0=lo;k0<hi;k0+=16){
    f8 s=(f8){0,0,0,0,0,0,0,0};
    #pragma unroll
    for(int c=0;c<NC;c++){
      h16 kb=*(const h16*)(base+(size_t)(k0+r16)*3*D+D+h*HD+c*16);
      s=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[c],kb,s);
    }
    { // stage this block's V (16 keys x HD) in LDS, coalesced: lane -> key lane/2, half the dims
      const int kk=lane>>1, part=lane&1;
      const _Float16* src=base+(size_t)(k0+kk)*3*D+2*D+h*HD+part*(HD/2);
      #pragma unroll
      for(int t=0;t<HD/2;t+=8) *(h8v*)&Vs[wave][kk][part*(HD/2)+t]=*(const h8v*)(src+t);
    }
    const int j=k0+r16;
    #pragma unroll
    for(int i=0;i<8;i++){
      const int q=q0+2*i+hl;
      const bool ok = j<valid && (window==0 || abs(q-j)<=window/2);
      const float v = ok ? s[i]*scale : -INFINITY;
      float mx=v;
      #pragma unroll
      for(int off=1;off<16;off<<=1) mx=fmaxf(mx,__shfl_xor(mx,off,16));
      const float mn=fmaxf(m[i],mx);
      float alpha, p;
      if(mn==-INFINITY){ alpha=1.f; p=0.f; }
      else { alpha=__expf(m[i]-mn); p= ok ? __expf(v-mn) : 0.f; }
      float rs=p;
      #pragma unroll
      for(int off=1;off<16;off<<=1) rs+=__shfl_xor(rs,off,16);
      l[i]=l[i]*alpha+rs; m[i]=mn;
      #pragma unroll
      for(int c=0;c<NC;c++) o[c][i]*=alpha;
      Ps[wave][2*i+hl][r16]=(_Float16)p;
    }
    __builtin_amdgcn_wave_barrier();
    const h16 pa=*(const h16*)&Ps[wave][r16][0];
    #pragma unroll
    for(int c=0;c<NC;c++){
      h16 vb;
      #pragma unroll
      for(int kk=0;kk<16;kk++) vb[kk]=Vs[wave][kk][c*16+r16];
      o[c]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa,vb,o[c]);
    }
    __builtin_amdgcn_wave_barrier();
  }
  #pragma unroll
  for(int i=0;i<8;i++){
    const int q=q0+2*i+hl;
    const float inv = l[i]>0.f ? 1.f/l[i] : 0.f;
    __half* dst=out+((size_t)soff[b]+q)*D+h*HD;
    #pragma unroll
    for(int c=0;c<NC;c++) dst[c*16+r16]=__float2half(o[c][i]*inv);
  }
}

// ---- GeGLU: out[r][c] = gelu(in[r][c]) * in[r][I+c], half2-vectorized (I is even). ----
__global__ void k_geglu(const __half* __restrict__ in, __half* __restrict__ out, int rows, int I){
  const int I2=I/2; const size_t n=(size_t)rows*I2;
  for(size_t idx=blockIdx.x*(size_t)blockDim.x+threadIdx.x; idx<n; idx+=(size_t)gridDim.x*blockDim.x){
    size_t row=idx/I2, c=idx-row*I2;
    float2 a=__half22float2(((const __half2*)(in+row*2*I))[c]);
    float2 g=__half22float2(((const __half2*)(in+row*2*I+I))[c]);
    ((__half2*)(out+row*I))[c]=__float22half2_rn(make_float2(gelu_erf(a.x)*g.x, gelu_erf(a.y)*g.y));
  }
}

// act: 0 = bias, 1 = bias+ReLU, 2 = bias+GELU
__global__ void k_bias_act(__half* __restrict__ x, const __half* __restrict__ b, int rows, int N, int act){
  const size_t n=(size_t)rows*N;
  for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){
    float v=__half2float(x[i])+__half2float(b[i%N]);
    if(act==1) v=fmaxf(v,0.f); else if(act==2) v=gelu_erf(v);
    x[i]=__float2half(v);
  }
}

template<typename T>
__global__ void k_gather(const T* __restrict__ h, const int* __restrict__ rows, T* __restrict__ m, int D){
  const int k=blockIdx.x; const size_t p=rows[k];
  for(int d=threadIdx.x;d<D;d+=blockDim.x) m[(size_t)k*D+d]=h[p*D+d];
}

__global__ void k_bias_to_float(const __half* __restrict__ x, const __half* __restrict__ b, float* __restrict__ y, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=__half2float(x[i])+__half2float(b[0]);
}

// ---- act head, one block (256 threads) per sequence. ----
// feats = [top1, top1-top2, normalized entropy, k/255] from softmax over the sequence's valid
// markers; hidden = GELU(W0 [pooled(CLS), feats] + b0); out = W2 hidden + b2.
__global__ __launch_bounds__(256) void k_act_head(const float* __restrict__ cls, int D,
    const float* __restrict__ logits, const int* __restrict__ moff, const int* __restrict__ nmark,
    const __half* __restrict__ W0, const __half* __restrict__ B0,
    const __half* __restrict__ W2, const __half* __restrict__ B2, int n_act, float* __restrict__ out){
  extern __shared__ float sm[];
  float* in=sm;            // D + 4
  float* hid=sm+D+4;       // 256
  const int b=blockIdx.x, t=threadIdx.x;
  for(int d=t;d<D;d+=256) in[d]=cls[(size_t)b*D+d];
  if(t==0){
    const float* z=logits+moff[b]; int K=nmark[b];
    float mx=-INFINITY; for(int i=0;i<K;i++) mx=fmaxf(mx,z[i]);
    float s=0.f; for(int i=0;i<K;i++) s+=__expf(z[i]-mx);
    float t1=0.f,t2=0.f,ent=0.f;
    for(int i=0;i<K;i++){ float p=__expf(z[i]-mx)/s; ent-=p*logf(fmaxf(p,1e-9f));
      if(p>t1){ t2=t1; t1=p; } else if(p>t2) t2=p; }
    float k=(float)(K<2?2:K);
    in[D]=t1; in[D+1]=t1-t2; in[D+2]=ent/logf(k); in[D+3]=k/255.f;
  }
  __syncthreads();
  { const __half* wr=W0+(size_t)t*(D+4); float a=__half2float(B0[t]);
    for(int i=0;i<D+4;i++) a+=__half2float(wr[i])*in[i];
    hid[t]=gelu_erf(a); }
  __syncthreads();
  if(t<n_act){ const __half* wr=W2+(size_t)t*256; float a=__half2float(B2[t]);
    for(int i=0;i<256;i++) a+=__half2float(wr[i])*hid[i];
    out[(size_t)b*n_act+t]=a; }
}

// transpose [R,C] -> [C,R] (weights arrive as [out,in]; NN GEMM wants [in,out])
__global__ void k_transpose(const __half* __restrict__ src,__half* __restrict__ dst,int R,int C){
  __shared__ __half tile[32][33];
  int bx=blockIdx.x*32, by=blockIdx.y*32;
  int x=bx+threadIdx.x, y=by+threadIdx.y;
  if(x<C && y<R) tile[threadIdx.y][threadIdx.x]=src[(size_t)y*C+x];
  __syncthreads();
  x=by+threadIdx.x; y=bx+threadIdx.y;
  if(x<R && y<C) dst[(size_t)y*R+x]=tile[threadIdx.x][threadIdx.y];
}

// ------------------------------------------------------------------ launchers
static inline int gs(size_t n){ size_t g=(n+255)/256; return (int)(g>1024?1024:(g?g:1)); }

template<typename Ti,typename To>
static void ln_(const Ti* x,const int* ids,const __half* w,const __half* b,To* y,int rows,int D,float eps,
                const __half* addtab,const int* addidx,const int* S,hipStream_t st){
  if(D%256||D>1024){ fprintf(stderr,"[laya] LayerNorm needs D%%256==0 && D<=1024 (D=%d)\n",D); abort(); }
  hipLaunchKernelGGL((k_ln<Ti,To>),dim3(rows),dim3(128),0,st,x,ids,w,b,y,D,eps,addtab,addidx,S); }
void launch_ln(const __half* x,const int* ids,const __half* w,const __half* b,__half* y,int rows,int D,float eps,const __half* at,const int* ai,const int* S,hipStream_t st){ ln_(x,ids,w,b,y,rows,D,eps,at,ai,S,st); }
void launch_ln(const __half* x,const int* ids,const __half* w,const __half* b,float* y,int rows,int D,float eps,const __half* at,const int* ai,const int* S,hipStream_t st){ ln_(x,ids,w,b,y,rows,D,eps,at,ai,S,st); }
void launch_ln(const float* x,const int* ids,const __half* w,const __half* b,__half* y,int rows,int D,float eps,const __half* at,const int* ai,const int* S,hipStream_t st){ ln_(x,ids,w,b,y,rows,D,eps,at,ai,S,st); }
void launch_ln(const float* x,const int* ids,const __half* w,const __half* b,float* y,int rows,int D,float eps,const __half* at,const int* ai,const int* S,hipStream_t st){ ln_(x,ids,w,b,y,rows,D,eps,at,ai,S,st); }
void launch_rope_inplace(__half* qkv,const float* c,const float* s,const int* pos,int rows,int H,int HD,hipStream_t st){
  hipLaunchKernelGGL(k_rope_inplace,dim3(rows),dim3(((H*HD/2+31)/32)*32),0,st,qkv,c,s,pos,H,HD); }
void launch_splitk_reduce(__half* dst,const __half* part,int splits,size_t n,int N,const __half* bias,int accumulate,hipStream_t st){
  hipLaunchKernelGGL((k_splitk_reduce<__half>),dim3(gs(n/2)),dim3(256),0,st,dst,part,splits,n,N,bias,accumulate); }
void launch_splitk_reduce(float* dst,const __half* part,int splits,size_t n,int N,const __half* bias,int accumulate,hipStream_t st){
  hipLaunchKernelGGL((k_splitk_reduce<float>),dim3(gs(n/2)),dim3(256),0,st,dst,part,splits,n,N,bias,accumulate); }
void launch_geglu(const __half* in,__half* out,int rows,int I,hipStream_t st){
  hipLaunchKernelGGL(k_geglu,dim3(gs((size_t)rows*I/2)),dim3(256),0,st,in,out,rows,I); }
void launch_bias_act(__half* x,const __half* b,int rows,int N,int act,hipStream_t st){
  hipLaunchKernelGGL(k_bias_act,dim3(gs((size_t)rows*N)),dim3(256),0,st,x,b,rows,N,act); }
void launch_gather(const float* h,const int* rows,float* m,int K,int D,hipStream_t st){
  hipLaunchKernelGGL((k_gather<float>),dim3(K),dim3(256),0,st,h,rows,m,D); }
void launch_gather(const __half* h,const int* rows,__half* m,int K,int D,hipStream_t st){
  hipLaunchKernelGGL((k_gather<__half>),dim3(K),dim3(256),0,st,h,rows,m,D); }
void launch_bias_to_float(const __half* x,const __half* b,float* y,int n,hipStream_t st){
  hipLaunchKernelGGL(k_bias_to_float,dim3((n+255)/256),dim3(256),0,st,x,b,y,n); }
void launch_act_head(const float* cls,int D,const float* logits,const int* moff,const int* nmark,
                     const __half* W0,const __half* B0,const __half* W2,const __half* B2,int n_act,float* out,int B,hipStream_t st){
  hipLaunchKernelGGL(k_act_head,dim3(B),dim3(256),(D+4+256)*sizeof(float),st,cls,D,logits,moff,nmark,W0,B0,W2,B2,n_act,out); }
void launch_flash_wmma(const __half* qkv,__half* out,int ntiles,int H,int HD,int window,const int* vlen,
                       const int* soff,const int* tseq,const int* tq0,float scale,hipStream_t st){
  dim3 g((ntiles+3)/4,H), blk(128);
  switch(HD){
    case 64:  hipLaunchKernelGGL((k_flash_wmma<64>), g,blk,0,st,qkv,out,ntiles,H,window,vlen,soff,tseq,tq0,scale); break;
    case 128: hipLaunchKernelGGL((k_flash_wmma<128>),g,blk,0,st,qkv,out,ntiles,H,window,vlen,soff,tseq,tq0,scale); break;
    case 32:  hipLaunchKernelGGL((k_flash_wmma<32>), g,blk,0,st,qkv,out,ntiles,H,window,vlen,soff,tseq,tq0,scale); break;
    default: fprintf(stderr,"[laya] flash: unsupported head_dim %d\n",HD); abort();
  }
}
void launch_res_ln(float* x,const __half* part,int splits,int rows,int D,const __half* bias,const __half* w,const __half* lb,
                   __half* y,float eps,const __half* at,const int* ai,const int* rs,hipStream_t st){
  hipLaunchKernelGGL((k_res_ln<__half>),dim3(rows),dim3(128),0,st,x,part,splits,(size_t)rows*D,bias,w,lb,y,D,eps,at,ai,rs); }
void launch_res_ln(float* x,const __half* part,int splits,int rows,int D,const __half* bias,const __half* w,const __half* lb,
                   float* y,float eps,const __half* at,const int* ai,const int* rs,hipStream_t st){
  hipLaunchKernelGGL((k_res_ln<float>),dim3(rows),dim3(128),0,st,x,part,splits,(size_t)rows*D,bias,w,lb,y,D,eps,at,ai,rs); }
void launch_transpose(const __half* s,__half* d,int R,int C,hipStream_t st){
  dim3 g((C+31)/32,(R+31)/32), b(32,32);
  hipLaunchKernelGGL(k_transpose,g,b,0,st,s,d,R,C); }
