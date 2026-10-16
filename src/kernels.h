#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
// LayerNorm overloads: residual stream is fp32, GEMM inputs are fp16
void launch_ln(const __half* x,const int* ids,const __half* w,const __half* b,__half* y,int rows,int D,float eps,const __half* at,const int* ai,const int* rowseq,hipStream_t st);
void launch_ln(const __half* x,const int* ids,const __half* w,const __half* b,float* y,int rows,int D,float eps,const __half* at,const int* ai,const int* rowseq,hipStream_t st);
void launch_ln(const float* x,const int* ids,const __half* w,const __half* b,__half* y,int rows,int D,float eps,const __half* at,const int* ai,const int* rowseq,hipStream_t st);
void launch_ln(const float* x,const int* ids,const __half* w,const __half* b,float* y,int rows,int D,float eps,const __half* at,const int* ai,const int* rowseq,hipStream_t st);
void launch_rope_inplace(__half* qkv,const float* c,const float* s,const int* pos,int rows,int H,int HD,hipStream_t st);
void launch_geglu(const __half* in,__half* out,int rows,int I,hipStream_t st);
void launch_bias_act(__half* x,const __half* b,int rows,int N,int act,hipStream_t st);   // 0 none, 1 ReLU, 2 GELU
void launch_gather(const float* h,const int* rows,float* m,int K,int D,hipStream_t st);
void launch_gather(const __half* h,const int* rows,__half* m,int K,int D,hipStream_t st);
void launch_bias_to_float(const __half* x,const __half* b,float* y,int n,hipStream_t st);
void launch_act_head(const float* cls,int D,const float* logits,int KP,const int* nmark,
                     const __half* W0,const __half* B0,const __half* W2,const __half* B2,int n_act,float* out,int B,hipStream_t st);
void launch_transpose(const __half* s,__half* d,int R,int C,hipStream_t st);
void launch_splitk_reduce(__half* dst,const __half* part,int splits,size_t n,int N,const __half* bias,int accumulate,hipStream_t st);
void launch_splitk_reduce(float* dst,const __half* part,int splits,size_t n,int N,const __half* bias,int accumulate,hipStream_t st);
// fused WMMA flash attention over packed sequences: qkv[rows][3D] -> out[rows][D]
void launch_flash_wmma(const __half* qkv,__half* out,int ntiles,int H,int HD,int window,const int* vlen,
                       const int* soff,const int* tseq,const int* tq0,float scale,hipStream_t st);
// residual update (x += sum of fp16 GEMM partials + bias) fused with the following LayerNorm -> y
void launch_res_ln(float* x,const __half* part,int splits,int rows,int D,const __half* bias,const __half* w,const __half* lb,
                   __half* y,float eps,const __half* at,const int* ai,const int* rs,hipStream_t st);
void launch_res_ln(float* x,const __half* part,int splits,int rows,int D,const __half* bias,const __half* w,const __half* lb,
                   float* y,float eps,const __half* at,const int* ai,const int* rs,hipStream_t st);
