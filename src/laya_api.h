#pragma once
#ifdef __cplusplus
extern "C" {
#endif
void* laya_create(const char* dir);
// B questions in one call. ids/markers are concatenated per question; markers are positions
// within that question's sequence. logits receive sum(nmark) floats, act B*n_act floats.
int   laya_forward_batch(void* h,int B,const int* lens,const int* ids,const int* nmark,const int* markers,
                         const int* qtypes,float* logits,float* act);
void  laya_forward(void* h,const int* ids,int S,int valid_len,const int* markers,int K,int qtype,
                   float* logits,float* act);
int   laya_n_act(void* h);
int   laya_max_len(void* h);
int   laya_max_markers(void* h);
float laya_temperature(void* h,int qtype);
void  laya_prof_report(void* h,int iters);
void  laya_prof_reset(void* h);
// Autotune every GEMM shape x M bucket up front (otherwise done lazily on first use).
void  laya_tune_all(void* h);
#ifdef __cplusplus
}
#endif
