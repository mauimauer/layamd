#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
// Custom WMMA GEMM: C[M,N] = A[M,K] * Wt[N,K]^T (Wt = weights in [out,in] layout).
// cfg 0: 128x128 tiles, cfg 1: 64x128 tiles. splits>1: fp16 partials C[z][M][N].
// geglu: Wt has 2I rows (input then gate); writes C[M,I] = gelu(A Wt_in^T) * (A Wt_gate^T).
int  wgemm_ncfg();
bool wgemm_supported(int cfg,int M,int N,int K,bool geglu,int I);
void launch_wgemm(int cfg,const __half* A,const __half* Wt,__half* C,int M,int N,int K,int splits,bool geglu,int I,hipStream_t st);
