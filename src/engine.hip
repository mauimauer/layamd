// Laya engine: weight upload + batched forward pass + per-shape HIP graphs.
//
// A pass runs many questions packed back to back: each sequence occupies its length rounded up
// to 16 rows (per-row position / sequence-id arrays drive RoPE and the type embedding; a tile
// map drives attention), and the pack is rounded up to 64 rows. FP16 GEMM inputs, FP32
// accumulation, FP32 residual stream. Attention is a fused WMMA flash kernel reading Q/K/V
// straight out of the fused qkv buffer (sliding-window layers only visit key blocks in band).
// The last head layer's out-projection and FFN run only on the rows the scorer / act head read:
// every question's option markers packed back to back (padded to 64 per pass), then one CLS row
// per question. Per-request inputs live in pinned host buffers at fixed addresses, so each
// (rows, sequences, marker rows) shape is captured once as a HIP graph and replayed afterwards.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <map>
#include <tuple>
#include <algorithm>
#include "laya.hpp"
#include "kernels.h"
#include "wgemm.h"

#define HC(x) do{auto e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP %s @%s:%d\n",hipGetErrorString(e),__FILE__,__LINE__);exit(1);} }while(0)
#define BC(x) do{auto e=(x); if(e!=HIPBLAS_STATUS_SUCCESS){fprintf(stderr,"BLAS %d @%s:%d\n",(int)e,__FILE__,__LINE__);exit(1);} }while(0)

namespace laya {

struct EncLayer { __half *attn_norm=nullptr,*Wqkv,*Wo,*mlp_norm,*Wi,*Wmo; };
struct HeadLayer { __half *n1w,*n1b,*n2w,*n2b,*inW,*inB,*outW,*outB,*l1W,*l1B,*l2W,*l2B; };

struct Prof {
  bool on=false; hipEvent_t a,b; std::vector<std::pair<std::string,float>> acc;
  void init(){ on=getenv("LAYA_PROFILE")!=nullptr; if(on){HC(hipEventCreate(&a));HC(hipEventCreate(&b));} }
  void s(hipStream_t st){ if(on) HC(hipEventRecord(a,st)); }
  void e(const char* n,hipStream_t st){ if(!on) return; HC(hipEventRecord(b,st)); HC(hipEventSynchronize(b));
    float ms; HC(hipEventElapsedTime(&ms,a,b));
    for(auto& kv:acc) if(kv.first==n){ kv.second+=ms; return; } acc.push_back({n,ms}); }
  void report(int iters){ if(!on) return; float tot=0; for(auto&kv:acc) tot+=kv.second;
    printf("\n%-18s %9s %7s\n","stage","ms/fwd","%");
    std::sort(acc.begin(),acc.end(),[](auto&x,auto&y){return x.second>y.second;});
    for(auto&kv:acc) printf("%-18s %9.3f %6.1f%%\n",kv.first.c_str(),kv.second/iters,100*kv.second/tot);
    printf("%-18s %9.3f (serialized sum)\n","TOTAL",tot/iters); acc.clear(); }
};
#define P(name, ...) do{ prof.s(stream); __VA_ARGS__; prof.e(name,stream); }while(0)

struct Engine {
  Config cfg; Prof prof;
  hipblasHandle_t blas; hipStream_t stream;
  std::vector<EncLayer> enc; std::vector<HeadLayer> head;
  __half *tok_emb,*emb_norm,*final_norm,*type_emb;
  __half *sc_nw,*sc_nb,*sc_1w,*sc_1b,*sc_3w,*sc_3b;
  __half *ah_0w,*ah_0b,*ah_2w,*ah_2b;
  float *cos_g,*sin_g,*cos_l,*sin_l;

  static constexpr int KMAX=255;      // options per question (the act head encodes the count as k/255)
  static constexpr int MAXB=128;      // sequences per pass
  int MAXTOK=8192, PAD=64;
  int pass_tokens=1024;   // rows per GPU pass: bigger packs spill the 32MB LLC (measured sweet spot ~1k)
  const char* dump_dir=getenv("LAYA_DUMP_DIR");
  bool use_graphs=true;


  float *x;                       // residual stream, fp32 (ModernBERT outliers reach ~3e4; fp16 step there is 16)
  __half *xn,*rtmp,*qkv_base,*qkv,*ao,*mi,*mg,*ff,*mk,*mk2;
  float *xg; __half *aog,*xng,*ffg;             // last head layer on gathered rows (markers + CLS)
  // split-K: fp16 partials + per-(M,N,K) split count autotuned on the first (eager) run of a shape
  __half *skw,*tune_dst; int max_splits=8, sk_maxM=1024, sk_maxN=0; bool capturing=false;
  // Per-(M,N,K) plan chosen by timing every candidate once: hipBLAS with split-K 1/2/4/8, or one
  // of hipBLASLt's heuristic algorithms (neither library wins on every shape on gfx1151).
  // wcfg >= 0: custom WMMA GEMM (wgemm.hip) on the [out,in] copy of W -- it wins on small-M,
  // N=1024 projections (attention-out, MLP-out) where library kernels leave CUs idle.
  struct Plan { int splits=1; bool lt=false; int wcfg=-1; hipblasLtMatrixLayout_t la,lb,lc; hipblasLtMatmulAlgo_t algo; };
  std::map<const __half*,__half*> worig;          // transposed weight -> original [out,in] copy
  // all weights sharing a GEMM shape (n,k): candidates are timed streaming through every one of
  // them, like the real forward pass does, instead of one cache-hot matrix
  std::map<std::pair<int,int>,std::vector<const __half*>> wset;
  std::pair<const __half*,int> partials(const Plan& p,const __half* A,const __half* W,int m,int n,int k);
  std::map<std::tuple<int,int,int>,Plan> tuned;   // chosen at a bucket M (see mbucket)
  std::map<std::tuple<int,int,int>,Plan> plans;   // concrete per actual M (own hipBLASLt layouts)
  __half* tune_a=nullptr;                          // scratch GEMM input for tuning at bucket M
  static int mbucket(int m){ if(m<=1024) return (m+63)/64*64;
    static const int g[]={1280,1536,1792,2048,2560,3072,3584,4096,5120,6144,7168,8192,10240,12288,14336,16384};
    for(int v:g) if(m<=v) return v; return (m+4095)/4096*4096; }
  const Plan& tune_bucket(const __half* W,int mb,int n,int k);
  int tune_cold_maxm=128;   // time candidates over all layers' (cold) weights up to this M
  void tune_all();
  hipblasLtHandle_t lt; hipblasLtMatmulDesc_t ltd; void* ltws; size_t ltwsz=32<<20; bool use_lt=true, use_wgemm=false;
  void gemm(const __half* A,const __half* W,__half* C,int m,int n,int k,bool accumulate,const __half* bias);
  void gemm_res(const __half* A,const __half* W,float* X,int m,int n,int k,const __half* bias);
  // gemm_res fused with the LayerNorm that follows it (y = LN(X) after the update)
  template<typename Tout>
  void gemm_res_ln(const __half* A,const __half* W,float* X,int m,int n,int k,const __half* bias,
                   const __half* lnw,const __half* lnb,Tout* y,const __half* at=nullptr,const int* ai=nullptr,const int* rs=nullptr);
  const Plan& plan_for(const __half* A,const __half* W,int m,int n,int k);
  void gemm_raw(const __half* A,const __half* W,__half* C,int m,int n,int k,bool accumulate,const __half* bias,const Plan& p);
  float *d_logits,*d_act;
  // per-pass inputs: device + pinned host mirrors (fixed addresses so graphs can replay them)
  int *d_ids,*d_pos,*d_seq,*d_soff,*d_vlen,*d_qt,*d_tseq,*d_tq0,*d_grow,*d_nm,*d_moff,*d_gtseq,*d_gdst;
  int *h_ids,*h_pos,*h_seq,*h_soff,*h_vlen,*h_qt,*h_tseq,*h_tq0,*h_grow,*h_nm,*h_moff,*h_gtseq,*h_gdst; float *h_logits,*h_act;
  std::map<std::tuple<int,int,int,int>,hipGraphExec_t> graphs;

  void load(const std::string& dir);
  __half* up(const Tensor& t);
  __half* upT(const Tensor& t,bool keep_orig=false);   // [out,in] -> [in,out] for NN GEMM (optionally keep the original)
  void run(int T,int B,int KT,int NTg);
  void execute(int T,int B,int KT,int NTg);
  int forward_batch(int B,const int* lens,const int* ids,const int* nmark,const int* markers,
                    const int* qtypes,float* logits,float* act);
};

__half* Engine::up(const Tensor& t){
  size_t cnt = t.dtype=="F32" ? t.nbytes/4 : t.nbytes/2;     // F16 / BF16: 2 bytes
  __half* d; HC(hipMalloc(&d,cnt*2));
  if(t.dtype=="F16") HC(hipMemcpy(d,t.data,cnt*2,hipMemcpyHostToDevice));
  else if(t.dtype=="F32" || t.dtype=="BF16"){        // fine-tunes are often saved in bf16
    std::vector<__half> tmp(cnt); float amax=0.f;
    for(size_t i=0;i<cnt;i++){
      float f; if(t.dtype=="F32") f=((const float*)t.data)[i];
      else { uint32_t u=(uint32_t)((const uint16_t*)t.data)[i]<<16; memcpy(&f,&u,4); }
      amax=fmaxf(amax,fabsf(f)); tmp[i]=__float2half(f); }
    if(amax>65504.f){ fprintf(stderr,"tensor exceeds fp16 range (|x| max %g)\n",amax); exit(1); }
    HC(hipMemcpy(d,tmp.data(),cnt*2,hipMemcpyHostToDevice)); }
  else { fprintf(stderr,"dtype %s unsupported\n",t.dtype.c_str()); exit(1); }
  return d;
}
__half* Engine::upT(const Tensor& t,bool keep_orig){
  __half* src=up(t);
  if(t.shape.size()!=2) return src;
  int R=(int)t.shape[0], C=(int)t.shape[1];
  __half* dst; HC(hipMalloc(&dst,(size_t)R*C*2));
  launch_transpose(src,dst,R,C,0); HC(hipDeviceSynchronize());
  if(keep_orig) worig[dst]=src; else HC(hipFree(src));
  return dst;
}

void Engine::load(const std::string& dir){
  cfg=Config::load(dir+"/encoder/config.json", dir+"/rl_agent_config.json");
  // custom WMMA GEMM: opt-in (measured no net gain vs hipBLAS/hipBLASLt); needs [out,in] weight copies
  use_wgemm = getenv("LAYA_WGEMM") && atoi(getenv("LAYA_WGEMM"))==1;
  cfg.dump();
  SafeTensors st;
  if(!st.open(dir+"/model.safetensors")) exit(1);
  tok_emb   = up(st.must("encoder.embeddings.tok_embeddings.weight"));
  emb_norm  = up(st.must("encoder.embeddings.norm.weight"));
  final_norm= up(st.must("encoder.final_norm.weight"));
  type_emb  = up(st.must("type_emb.weight"));
  enc.resize(cfg.n_layers);
  for(int i=0;i<cfg.n_layers;i++){
    char b[128]; auto T=[&](const char* f){ snprintf(b,sizeof b,f,i); return std::string(b); };
    auto opt=st.get(T("encoder.layers.%d.attn_norm.weight"));
    enc[i].attn_norm = opt? up(*opt) : nullptr;       // layer 0 has none (Identity)
    enc[i].Wqkv = upT(st.must(T("encoder.layers.%d.attn.Wqkv.weight")));
    enc[i].Wo   = upT(st.must(T("encoder.layers.%d.attn.Wo.weight")),use_wgemm);
    enc[i].mlp_norm=up(st.must(T("encoder.layers.%d.mlp_norm.weight")));
    enc[i].Wi   = upT(st.must(T("encoder.layers.%d.mlp.Wi.weight")));
    enc[i].Wmo  = upT(st.must(T("encoder.layers.%d.mlp.Wo.weight")),use_wgemm);
  }
  head.resize(cfg.head_layers);
  for(int i=0;i<cfg.head_layers;i++){
    char b[128]; auto T=[&](const char* f){ snprintf(b,sizeof b,f,i); return std::string(b); };
    head[i].n1w=up(st.must(T("head.layers.%d.norm1.weight"))); head[i].n1b=up(st.must(T("head.layers.%d.norm1.bias")));
    head[i].n2w=up(st.must(T("head.layers.%d.norm2.weight"))); head[i].n2b=up(st.must(T("head.layers.%d.norm2.bias")));
    head[i].inW=upT(st.must(T("head.layers.%d.self_attn.in_proj_weight"))); head[i].inB=up(st.must(T("head.layers.%d.self_attn.in_proj_bias")));
    head[i].outW=upT(st.must(T("head.layers.%d.self_attn.out_proj.weight")),use_wgemm); head[i].outB=up(st.must(T("head.layers.%d.self_attn.out_proj.bias")));
    head[i].l1W=upT(st.must(T("head.layers.%d.linear1.weight"))); head[i].l1B=up(st.must(T("head.layers.%d.linear1.bias")));
    head[i].l2W=upT(st.must(T("head.layers.%d.linear2.weight")),use_wgemm); head[i].l2B=up(st.must(T("head.layers.%d.linear2.bias")));
  }
  sc_nw=up(st.must("scorer.0.weight")); sc_nb=up(st.must("scorer.0.bias"));
  sc_1w=upT(st.must("scorer.1.weight")); sc_1b=up(st.must("scorer.1.bias"));
  sc_3w=upT(st.must("scorer.3.weight")); sc_3b=up(st.must("scorer.3.bias"));
  ah_0w=up(st.must("act_head.0.weight")); ah_0b=up(st.must("act_head.0.bias"));
  ah_2w=up(st.must("act_head.2.weight")); ah_2b=up(st.must("act_head.2.bias"));

  { const int D=cfg.d,I=cfg.intermediate,FF=cfg.head_ffn;
    for(auto& l:enc){ wset[{3*D,D}].push_back(l.Wqkv); wset[{D,D}].push_back(l.Wo); wset[{2*I,D}].push_back(l.Wi); wset[{D,I}].push_back(l.Wmo); }
    for(auto& h:head){ wset[{3*D,D}].push_back(h.inW); wset[{D,D}].push_back(h.outW); wset[{FF,D}].push_back(h.l1W); wset[{D,FF}].push_back(h.l2W); }
    wset[{D,D}].push_back(sc_1w); wset[{1,D}].push_back(sc_3w); }
  if(const char* s=getenv("LAYA_TUNE_COLD_MAXM")) tune_cold_maxm=atoi(s);
  int H2=cfg.head_dim/2, L=cfg.max_len;
  std::vector<float> cg(L*H2),sg(L*H2),cl(L*H2),sl(L*H2);
  for(int p=0;p<L;p++) for(int i=0;i<H2;i++){
    float fg=p/powf(cfg.rope_theta_global,(2.f*i)/cfg.head_dim);
    float fl=p/powf(cfg.rope_theta_local ,(2.f*i)/cfg.head_dim);
    cg[p*H2+i]=cosf(fg); sg[p*H2+i]=sinf(fg); cl[p*H2+i]=cosf(fl); sl[p*H2+i]=sinf(fl);
  }
  auto upf=[&](std::vector<float>& v){ float* d; HC(hipMalloc(&d,v.size()*4)); HC(hipMemcpy(d,v.data(),v.size()*4,hipMemcpyHostToDevice)); return d; };
  cos_g=upf(cg); sin_g=upf(sg); cos_l=upf(cl); sin_l=upf(sl);

  if(const char* e=getenv("LAYA_MAX_TOKENS")) MAXTOK=std::max(atoi(e),cfg.max_len);
  MAXTOK=std::max(MAXTOK,cfg.max_len);
  if(const char* e=getenv("LAYA_PASS_TOKENS")) pass_tokens=atoi(e);
  pass_tokens=std::min(std::max(pass_tokens,cfg.max_len),MAXTOK);
  use_graphs = !(getenv("LAYA_GRAPHS") && atoi(getenv("LAYA_GRAPHS"))==0);

  BC(hipblasCreate(&blas)); HC(hipStreamCreate(&stream)); BC(hipblasSetStream(blas,stream));
  prof.init(); if(prof.on) use_graphs=false;    // per-op event syncs cannot be captured


  const int D=cfg.d,I=cfg.intermediate,FF=cfg.head_ffn, MT=MAXTOK;
  const int maxB=MAXB, RG=MT+MAXB, maxNTg=MT/16+MAXB; // gathered rows: <= MT marker rows + one CLS row per sequence
  auto A=[&](size_t n){ __half* p; HC(hipMalloc(&p,n*2)); HC(hipMemset(p,0,n*2)); return p; };
  HC(hipMalloc(&x,(size_t)MT*D*4)); xn=A((size_t)MT*D); ao=A((size_t)MT*D); rtmp=A((size_t)MT*D);
  qkv_base=A((size_t)(PAD+MT+PAD)*3*D); qkv=qkv_base+(size_t)PAD*3*D;
  mi=A((size_t)MT*2*I); mg=A((size_t)MT*I); ff=A((size_t)MT*FF);
  mk=A((size_t)MT*D); mk2=A((size_t)MT*D);
  HC(hipMalloc(&xg,(size_t)RG*D*4)); aog=A((size_t)RG*D); xng=A((size_t)RG*D); ffg=A((size_t)RG*FF);
  if(const char* s=getenv("LAYA_SPLITK")) max_splits=std::max(1,atoi(s));
  sk_maxN=std::max(std::max(2*I,FF),3*D);
  HC(hipMalloc(&skw,(size_t)max_splits*sk_maxM*sk_maxN*2));
  HC(hipMalloc(&tune_dst,(size_t)std::max(mbucket(MT),RG)*sk_maxN*2));
  HC(hipMalloc(&tune_a,(size_t)mbucket(MT)*std::max(std::max(I,FF),D)*2));
  HC(hipMemset(tune_a,0,(size_t)mbucket(MT)*std::max(std::max(I,FF),D)*2));
  use_lt = !(getenv("LAYA_HIPBLASLT") && atoi(getenv("LAYA_HIPBLASLT"))==0);
  if(use_lt){
    if(hipblasLtCreate(&lt)!=HIPBLAS_STATUS_SUCCESS) use_lt=false;
    else { BC(hipblasLtMatmulDescCreate(&ltd,HIPBLAS_COMPUTE_32F,HIP_R_32F));
      hipblasOperation_t op=HIPBLAS_OP_N;
      BC(hipblasLtMatmulDescSetAttribute(ltd,HIPBLASLT_MATMUL_DESC_TRANSA,&op,sizeof op));
      BC(hipblasLtMatmulDescSetAttribute(ltd,HIPBLASLT_MATMUL_DESC_TRANSB,&op,sizeof op));
      HC(hipMalloc(&ltws,ltwsz)); }
  }
  HC(hipMalloc(&d_logits,(size_t)MT*4)); HC(hipMalloc(&d_act,(size_t)maxB*8*4));
  HC(hipHostMalloc(&h_logits,(size_t)MT*4)); HC(hipHostMalloc(&h_act,(size_t)maxB*8*4));
  struct IA{ int** d; int** h; size_t n; } ia[]={
    {&d_ids,&h_ids,(size_t)MT},{&d_pos,&h_pos,(size_t)MT},{&d_seq,&h_seq,(size_t)MT},
    {&d_soff,&h_soff,(size_t)maxB+1},{&d_vlen,&h_vlen,(size_t)maxB+1},{&d_qt,&h_qt,(size_t)maxB+1},
    {&d_tseq,&h_tseq,(size_t)MT/16},{&d_tq0,&h_tq0,(size_t)MT/16},{&d_grow,&h_grow,(size_t)RG},
    {&d_nm,&h_nm,(size_t)maxB},{&d_moff,&h_moff,(size_t)maxB+1},
    {&d_gtseq,&h_gtseq,(size_t)maxNTg},{&d_gdst,&h_gdst,(size_t)maxNTg*16}};
  for(auto& a:ia){ HC(hipMalloc(a.d,a.n*4)); HC(hipHostMalloc(a.h,a.n*4)); }
  printf("[laya] ready: max_tokens/pass=%d graphs=%d\n",MAXTOK,(int)use_graphs);
}

// Run plan p producing fp16 output for a following reduce: returns (buffer, #partials).
std::pair<const __half*,int> Engine::partials(const Plan& p,const __half* A,const __half* W,int m,int n,int k){
  const float one=1.f, zero=0.f;
  if(p.wcfg>=0){
    __half* out = p.splits>1 ? skw : rtmp;
    launch_wgemm(p.wcfg,A,worig.at(W),out,m,n,k,p.splits,false,0,stream);
    return {out,p.splits};
  }
  if(!p.lt && p.splits>1){
    const int kc=k/p.splits;
    BC(hipblasGemmStridedBatchedEx(blas,HIPBLAS_OP_N,HIPBLAS_OP_N,n,m,kc,&one,
        W,HIP_R_16F,n,(long long)kc*n, A,HIP_R_16F,k,(long long)kc, &zero,
        skw,HIP_R_16F,n,(long long)m*n, p.splits,HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT));
    return {skw,p.splits};
  }
  Plan q=p; q.splits=1;
  gemm_raw(A,W,rtmp,m,n,k,false,nullptr,q);
  return {rtmp,1};
}

void Engine::gemm_raw(const __half* A,const __half* W,__half* C,int m,int n,int k,bool accumulate,const __half* bias,const Plan& p){
  const float one=1.f, zero=0.f;
  if(p.wcfg>=0){
    if(p.splits<=1 && !accumulate){ launch_wgemm(p.wcfg,A,worig.at(W),C,m,n,k,1,false,0,stream);
      if(bias) launch_bias_act(C,bias,m,n,0,stream); return; }
    auto pr=partials(p,A,W,m,n,k);
    launch_splitk_reduce(C,pr.first,pr.second,(size_t)m*n,n,bias,accumulate?1:0,stream);
    return;
  }
  // row-major C[m,n] = A[m,k] W[k,n]  ==  column-major C^T[n,m] = W^T[n,k] A^T[k,m]
  if(p.lt){
    BC(hipblasLtMatmul(lt,ltd,&one,W,p.la,A,p.lb,accumulate?&one:&zero,C,p.lc,C,p.lc,&p.algo,ltws,ltwsz,stream));
    if(bias) launch_bias_act(C,bias,m,n,0,stream);
    return;
  }
  if(p.splits<=1){
    BC(hipblasGemmEx(blas,HIPBLAS_OP_N,HIPBLAS_OP_N,n,m,k,&one,W,HIP_R_16F,n,A,HIP_R_16F,k,
        accumulate?&one:&zero,C,HIP_R_16F,n,HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT));
    if(bias) launch_bias_act(C,bias,m,n,0,stream);
    return;
  }
  const int kc=k/p.splits;
  BC(hipblasGemmStridedBatchedEx(blas,HIPBLAS_OP_N,HIPBLAS_OP_N,n,m,kc,&one,
      W,HIP_R_16F,n,(long long)kc*n, A,HIP_R_16F,k,(long long)kc, &zero,
      skw,HIP_R_16F,n,(long long)m*n, p.splits,HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT));
  launch_splitk_reduce(C,skw,p.splits,(size_t)m*n,n,bias,accumulate?1:0,stream);
}

template<typename Tout>
void Engine::gemm_res_ln(const __half* A,const __half* W,float* X,int m,int n,int k,const __half* bias,
                         const __half* lnw,const __half* lnb,Tout* y,const __half* at,const int* ai,const int* rs){
  auto pr=partials(plan_for(A,W,m,n,k),A,W,m,n,k);
  launch_res_ln(X,pr.first,pr.second,m,n,bias,lnw,lnb,y,cfg.norm_eps,at,ai,rs,stream);
}

// Time every candidate for (mb,n,k) on scratch buffers: hipBLAS with split-K 1/2/4/8 and the top
// hipBLASLt heuristic algorithms. Tuning happens per M *bucket*, so dynamic batch sizes don't
// trigger new tuning on the request path (tune_all() covers every bucket at startup).
const Engine::Plan& Engine::tune_bucket(const __half* W,int mb,int n,int k){
  auto key=std::make_tuple(mb,n,k);
  auto it=tuned.find(key); if(it!=tuned.end()) return it->second;
  Plan best;
  if(!capturing && n<=sk_maxN){
    std::vector<Plan> cands;
    for(int sp=1;sp<=max_splits;sp*=2){
      if(sp>1 && (mb>sk_maxM || (n&1) || k%sp || (k/sp)%8)) continue;
      Plan p; p.splits=sp; cands.push_back(p);
    }
    if(use_wgemm && worig.count(W))
      for(int c=0;c<wgemm_ncfg();c++) for(int sp=1;sp<=max_splits;sp*=2){
        if(!wgemm_supported(c,mb,n,k,false,0) || (sp>1 && mb>sk_maxM) || k/64<sp) continue;
        Plan p; p.wcfg=c; p.splits=sp; cands.push_back(p);
      }
    if(use_lt){
      Plan p; p.lt=true;
      BC(hipblasLtMatrixLayoutCreate(&p.la,HIP_R_16F,n,k,n));
      BC(hipblasLtMatrixLayoutCreate(&p.lb,HIP_R_16F,k,mb,k));
      BC(hipblasLtMatrixLayoutCreate(&p.lc,HIP_R_16F,n,mb,n));
      hipblasLtMatmulPreference_t pr; BC(hipblasLtMatmulPreferenceCreate(&pr));
      BC(hipblasLtMatmulPreferenceSetAttribute(pr,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ltwsz,sizeof ltwsz));
      hipblasLtMatmulHeuristicResult_t hr[8]; int nh=0;
      if(hipblasLtMatmulAlgoGetHeuristic(lt,ltd,p.la,p.lb,p.lc,p.lc,pr,8,hr,&nh)==HIPBLAS_STATUS_SUCCESS)
        for(int i=0;i<nh;i++){ p.algo=hr[i].algo; cands.push_back(p); }
      hipblasLtMatmulPreferenceDestroy(pr);
    }
    HC(hipStreamSynchronize(stream));
    hipEvent_t a,b; HC(hipEventCreate(&a)); HC(hipEventCreate(&b));
    float bt=1e30f; const Plan* win=&cands[0];
    // Small M is DRAM-bound in the real pass (28 layers of distinct weights stream from memory),
    // so there candidates are timed across every weight of this shape; larger M is compute-bound
    // and timed on one (cache-hot) matrix. Only weights with an [out,in] copy qualify for wgemm.
    std::vector<const __half*> ws;
    auto it_ws=wset.find({n,k});
    if(mb<=tune_cold_maxm && it_ws!=wset.end()) for(auto w:it_ws->second) if(w==W || !worig.count(W) || worig.count(w)) ws.push_back(w);
    if(ws.empty()) ws.push_back(W);
    const int reps = ws.size()>1 ? 1 : (mb>=2048 ? 3 : 5);
    for(auto& p:cands){
      const float one=1.f, zero=0.f;
      if(p.lt && hipblasLtMatmul(lt,ltd,&one,W,p.la,tune_a,p.lb,&zero,tune_dst,p.lc,tune_dst,p.lc,&p.algo,ltws,ltwsz,stream)!=HIPBLAS_STATUS_SUCCESS) continue;
      for(auto w:ws) gemm_raw(tune_a,w,tune_dst,mb,n,k,false,nullptr,p);
      HC(hipEventRecord(a,stream));
      for(int r=0;r<reps;r++) for(auto w:ws) gemm_raw(tune_a,w,tune_dst,mb,n,k,false,nullptr,p);
      HC(hipEventRecord(b,stream)); HC(hipEventSynchronize(b));
      float ms; HC(hipEventElapsedTime(&ms,a,b)); ms/=reps*(int)ws.size();
      if(ms<bt*0.97f){ bt=ms; win=&p; }
    }
    HC(hipEventDestroy(a)); HC(hipEventDestroy(b));
    best=*win;
    if(getenv("LAYA_VERBOSE")) printf("[laya] tune %5dx%5dx%5d -> %s%d  %.1f us\n",mb,n,k,best.lt?"hipblaslt":"hipblas split ",best.lt?0:best.splits,bt*1000);
  }
  return tuned.emplace(key,best).first->second;
}

const Engine::Plan& Engine::plan_for(const __half* A,const __half* W,int m,int n,int k){
  auto key=std::make_tuple(m,n,k);
  auto it=plans.find(key); if(it!=plans.end()) return it->second;
  const int mb=std::min(mbucket(m),mbucket(MAXTOK));
  Plan p=tune_bucket(W,mb,n,k);
  if(p.splits>1 && m>sk_maxM) p.splits=1;
  if(p.wcfg>=0 && !wgemm_supported(p.wcfg,m,n,k,false,0)){ p.wcfg=-1; p.splits=1; }
  if(p.lt){                                          // same algorithm, layouts for the actual M
    BC(hipblasLtMatrixLayoutCreate(&p.la,HIP_R_16F,n,k,n));
    BC(hipblasLtMatrixLayoutCreate(&p.lb,HIP_R_16F,k,m,k));
    BC(hipblasLtMatrixLayoutCreate(&p.lc,HIP_R_16F,n,m,n));
    const float one=1.f, zero=0.f;
    if(capturing || hipblasLtMatmul(lt,ltd,&one,W,p.la,A,p.lb,&zero,tune_dst,p.lc,tune_dst,p.lc,&p.algo,ltws,ltwsz,stream)!=HIPBLAS_STATUS_SUCCESS)
      p.lt=false, p.splits=1;                        // algorithm refuses this M: plain hipBLAS
  }
  return plans.emplace(key,p).first->second;
}

// Tune every (M bucket, GEMM shape) the model can hit, so serving never tunes on the request path.
void Engine::tune_all(){
  const int D=cfg.d,I=cfg.intermediate,FF=cfg.head_ffn;
  struct Sh{ const __half* W; int n,k; };
  std::vector<Sh> sh={{enc[0].Wqkv,3*D,D},{enc[0].Wo,D,D},{enc[0].Wi,2*I,D},{enc[0].Wmo,D,I},
                      {head[0].l1W,FF,D},{head[0].l2W,D,FF},{sc_3w,1,D}};
  // every M bucket up to pass_tokens: covers the encoder and, since marker+CLS rows are packed
  // (<= pass_tokens), the gathered head rows and the scorer too -- they share these (n,k) shapes
  for(int mb=64; mb<=mbucket(pass_tokens); mb=mbucket(mb+1))
    for(auto& s:sh) tune_bucket(s.W,mb,s.n,s.k);
}

// C[m,n] (+)= A[m,k] @ W[k,n] (+ bias). W is pre-transposed to [in,out].
void Engine::gemm(const __half* A,const __half* W,__half* C,int m,int n,int k,bool accumulate,const __half* bias){
  gemm_raw(A,W,C,m,n,k,accumulate,bias,plan_for(A,W,m,n,k));
}

// X[m,n] (fp32 residual) += A[m,k] @ W[k,n] (+ bias). The GEMM writes fp16 (split-K partials or
// one plain output); the reduce kernel adds it -- and the bias -- into the fp32 stream.
void Engine::gemm_res(const __half* A,const __half* W,float* X,int m,int n,int k,const __half* bias){
  auto pr=partials(plan_for(A,W,m,n,k),A,W,m,n,k);
  launch_splitk_reduce(X,pr.first,pr.second,(size_t)m*n,n,bias,1,stream);
}


// Enqueue one full forward for the pack currently staged in the pinned host buffers:
// T rows (multiple of 64), B sequences, KT packed marker+CLS rows (multiple of 64).
void Engine::run(int T,int B,int KT,int NTg){
  const int D=cfg.d,H=cfg.n_heads,HD=cfg.head_dim,I=cfg.intermediate,FF=cfg.head_ffn, M=T, NT=T/16, R=KT+B;
  const float eps=cfg.norm_eps, scale=1.f/sqrtf((float)HD);
  auto linear=[&](const __half* A_,const __half* W,__half* C,int m,int n,int k){ gemm(A_,W,C,m,n,k,false,nullptr); };
  auto attend=[&](int window){ launch_flash_wmma(qkv,ao,NT,H,HD,window,d_vlen,d_soff,d_tseq,d_tq0,scale,stream); };
  auto h2d=[&](int* d,const int* h,size_t n){ HC(hipMemcpyAsync(d,h,n*4,hipMemcpyHostToDevice,stream)); };

  h2d(d_ids,h_ids,T); h2d(d_pos,h_pos,T); h2d(d_seq,h_seq,T);
  h2d(d_soff,h_soff,B+1); h2d(d_vlen,h_vlen,B+1); h2d(d_qt,h_qt,B+1);
  h2d(d_tseq,h_tseq,NT); h2d(d_tq0,h_tq0,NT); h2d(d_grow,h_grow,R);
  h2d(d_nm,h_nm,B); h2d(d_moff,h_moff,B);
  h2d(d_gtseq,h_gtseq,NTg); h2d(d_gdst,h_gdst,(size_t)NTg*16);

  // embeddings -> fp32 residual x, plus the same values in fp16 for layer 0 (no attn_norm there)
  P("enc:layernorm", launch_ln(tok_emb,d_ids,emb_norm,nullptr,x,M,D,eps,nullptr,nullptr,nullptr,stream,enc[0].attn_norm?nullptr:xn));
  // Each residual update is fused with the LayerNorm that follows it (next layer's attn_norm, or
  // the final norm + per-sequence type embedding written back into x).
  for(int L=0;L<cfg.n_layers;L++){
    const bool glob = cfg.layer_types[L]==AttnKind::Global;
    if(L==0 && enc[0].attn_norm) P("enc:layernorm", launch_ln(x,nullptr,enc[0].attn_norm,nullptr,xn,M,D,eps,nullptr,nullptr,nullptr,stream));
    P("enc:gemm_qkv", linear(xn,enc[L].Wqkv,qkv,M,3*D,D));
    P("enc:rope", launch_rope_inplace(qkv,glob?cos_g:cos_l,glob?sin_g:sin_l,d_pos,M,H,HD,stream));
    P(glob?"enc:attn_global":"enc:attn_slide", attend(glob?0:cfg.local_attention));
    P("enc:gemm_attn_out+ln", gemm_res_ln(ao,enc[L].Wo,x,M,D,D,nullptr,enc[L].mlp_norm,(const __half*)nullptr,xn));
    P("enc:gemm_mlp_in", linear(xn,enc[L].Wi,mi,M,2*I,D));
    P("enc:geglu", launch_geglu(mi,mg,M,I,stream));
    if(L+1<cfg.n_layers){
      if(enc[L+1].attn_norm) P("enc:gemm_mlp_out+ln", gemm_res_ln(mg,enc[L].Wmo,x,M,D,I,nullptr,enc[L+1].attn_norm,(const __half*)nullptr,xn));
      else                   P("enc:gemm_mlp_out", gemm_res(mg,enc[L].Wmo,x,M,D,I,nullptr));
    } else P("enc:gemm_mlp_out+ln", gemm_res_ln(mg,enc[L].Wmo,x,M,D,I,nullptr,final_norm,(const __half*)nullptr,x,type_emb,d_qt,d_seq));
    if(dump_dir && !capturing){                           // debug: residual stream after each layer
      HC(hipStreamSynchronize(stream)); std::vector<float> fb((size_t)M*D);
      HC(hipMemcpy(fb.data(),x,fb.size()*4,hipMemcpyDeviceToHost));
      char fn[512]; snprintf(fn,sizeof fn,"%s/enc%d.bin",dump_dir,L);
      FILE* f=fopen(fn,"wb"); fwrite(fb.data(),4,fb.size(),f); fclose(f);
    }
  }
  for(int L=0;L<cfg.head_layers;L++){
    const HeadLayer& hl=head[L];
    const bool last = L==cfg.head_layers-1;
    if(L==0) P("head", launch_ln(x,nullptr,hl.n1w,hl.n1b,xn,M,D,eps,nullptr,nullptr,nullptr,stream));
    P("head", linear(xn,hl.inW,qkv,M,3*D,D);
              launch_bias_act(qkv,hl.inB,M,3*D,0,stream);
              if(!last) attend(0));
    if(!last){
      const HeadLayer& nx=head[L+1];                        // not last, so a next layer exists
      P("head", gemm_res_ln(ao,hl.outW,x,M,D,D,hl.outB,hl.n2w,hl.n2b,xn);
                linear(xn,hl.l1W,ff,M,FF,D);
                launch_bias_act(ff,hl.l1B,M,FF,1,stream);
                gemm_res_ln(ff,hl.l2W,x,M,D,FF,hl.l2B,nx.n1w,nx.n1b,xn));
    } else {
      // only marker rows (scorer) and CLS rows (act head) are read after this layer:
      // attention queries, out-proj and FFN run only on those R = KT+B rows (KT packed marker rows, then B CLS rows)
      P("head", launch_flash_wmma(qkv,aog,NTg,H,HD,0,d_vlen,d_soff,d_gtseq,d_gdst,scale,stream,d_grow);
                launch_gather(x,d_grow,xg,R,D,stream);
                gemm_res_ln(aog,hl.outW,xg,R,D,D,hl.outB,hl.n2w,hl.n2b,xng);
                linear(xng,hl.l1W,ffg,R,FF,D);
                launch_bias_act(ffg,hl.l1B,R,FF,1,stream);
                gemm_res(ffg,hl.l2W,xg,R,D,FF,hl.l2B));
    }
  }
  const float* rows = cfg.head_layers>0 ? xg : x;       // (head_layers==0 is not produced by upstream)
  P("scorer", launch_ln(rows,nullptr,sc_nw,sc_nb,mk2,KT,D,eps,nullptr,nullptr,nullptr,stream);
              linear(mk2,sc_1w,mk,KT,D,D);
              launch_bias_act(mk,sc_1b,KT,D,2,stream);
              linear(mk,sc_3w,mk2,KT,1,D);
              launch_bias_to_float(mk2,sc_3b,d_logits,KT,stream);
              launch_act_head(rows+(size_t)KT*D,D,d_logits,d_moff,d_nm,ah_0w,ah_0b,ah_2w,ah_2b,cfg.n_act,d_act,B,stream));
  HC(hipMemcpyAsync(h_logits,d_logits,(size_t)KT*4,hipMemcpyDeviceToHost,stream));
  HC(hipMemcpyAsync(h_act,d_act,(size_t)B*cfg.n_act*4,hipMemcpyDeviceToHost,stream));
}

void Engine::execute(int T,int B,int KT,int NTg){
  auto key=std::make_tuple(T,B,KT,NTg);
  if(use_graphs){
    auto it=graphs.find(key);
    if(it!=graphs.end()){ HC(hipGraphLaunch(it->second,stream)); HC(hipStreamSynchronize(stream)); return; }
  }
  run(T,B,KT,NTg);                             // eager: produces this call's result
  HC(hipStreamSynchronize(stream));
  if(!use_graphs || graphs.size()>=512) return;
  hipGraph_t g; hipGraphExec_t ex;
  if(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal)!=hipSuccess){ use_graphs=false; return; }
  capturing=true; run(T,B,KT,NTg); capturing=false;
  if(hipStreamEndCapture(stream,&g)!=hipSuccess || hipGraphInstantiate(&ex,g,nullptr,nullptr,0)!=hipSuccess){
    fprintf(stderr,"[laya] graph capture failed; continuing without graphs\n");
    (void)hipGetLastError(); use_graphs=false; return;
  }
  HC(hipGraphDestroy(g));
  graphs[key]=ex;
}

// Pack questions back to back (16-row aligned), MAXTOK rows / MAXB sequences per pass.
int Engine::forward_batch(int B,const int* lens,const int* ids,const int* nmark,const int* markers,
                          const int* qtypes,float* logits,float* act){
  std::vector<size_t> ioff(B+1,0), moff(B+1,0);
  for(int b=0;b<B;b++){
    if(lens[b]<1 || lens[b]>cfg.max_len || nmark[b]<1 || nmark[b]>KMAX) return -1;
    ioff[b+1]=ioff[b]+lens[b]; moff[b+1]=moff[b]+nmark[b];
    for(int k=0;k<nmark[b];k++) if(markers[moff[b]+k]<0 || markers[moff[b]+k]>=lens[b]) return -1;
  }
  int b0=0;
  while(b0<B){
    int n=0, T=0, nk=0;
    while(b0+n<B && n<MAXB){
      int l16=(lens[b0+n]+15)/16*16;
      if(T+l16>pass_tokens) break;
      T+=l16; nk+=nmark[b0+n]; n++;
    }
    const int Tp=(T+63)/64*64, KT=(nk+63)/64*64;
    int r=0, mo=0, gt=0;
    for(int i=0;i<n;i++){
      const int b=b0+i, len=lens[b], l16=(len+15)/16*16, nm=nmark[b], nq=nm+1, nt=(nq+15)/16;
      h_soff[i]=r; h_vlen[i]=len; h_qt[i]=qtypes[b]; h_nm[i]=nm;
      memcpy(h_ids+r,ids+ioff[b],len*4);
      for(int j=len;j<l16;j++) h_ids[r+j]=cfg.pad_id;
      for(int j=0;j<l16;j++){ h_pos[r+j]=j; h_seq[r+j]=i; }
      for(int t=0;t<l16/16;t++){ h_tseq[r/16+t]=i; h_tq0[r/16+t]=t*16; }
      h_moff[i]=mo;
      for(int k=0;k<nm;k++) h_grow[mo+k]=r+markers[moff[b]+k];
      h_grow[KT+i]=r;                                // CLS row
      for(int t=0;t<nt;t++){
        h_gtseq[gt+t]=i;
        for(int lane=0;lane<16;lane++){
          int qidx=t*16+lane;
          h_gdst[(gt+t)*16+lane] = qidx<nm ? (mo+qidx) : (qidx==nm ? (KT+i) : -1);
        }
      }
      mo+=nm; gt+=nt; r+=l16;
    }
    for(int k=mo;k<KT;k++) h_grow[k]=0;              // padding rows: computed, never read
    for(int j=r;j<Tp;j++){ h_ids[j]=cfg.pad_id; h_pos[j]=0; h_seq[j]=n; }   // tail rows -> dummy seq n
    for(int t=r/16;t<Tp/16;t++){ h_tseq[t]=-1; h_tq0[t]=0; }
    h_soff[n]=r; h_vlen[n]=0; h_qt[n]=0;
    execute(Tp,n,KT,gt);
    for(int i=0;i<n;i++){
      int b=b0+i;
      memcpy(logits+moff[b],h_logits+h_moff[i],nmark[b]*4);
      memcpy(act+(size_t)b*cfg.n_act,h_act+(size_t)i*cfg.n_act,cfg.n_act*4);
    }
    b0+=n;
  }
  return 0;
}

} // namespace laya

// ---------------- C API ----------------
extern "C" {
void* laya_create(const char* dir){ auto* e=new laya::Engine(); e->load(dir); return e; }
int laya_forward_batch(void* h,int B,const int* lens,const int* ids,const int* nmark,const int* markers,
                       const int* qtypes,float* logits,float* act){
  return ((laya::Engine*)h)->forward_batch(B,lens,ids,nmark,markers,qtypes,logits,act); }
void laya_forward(void* h,const int* ids,int S,int valid_len,const int* markers,int K,int qtype,
                  float* logits,float* act){
  (void)S; ((laya::Engine*)h)->forward_batch(1,&valid_len,ids,&K,markers,&qtype,logits,act); }
int   laya_n_act(void* h){ return ((laya::Engine*)h)->cfg.n_act; }
int   laya_max_len(void* h){ return ((laya::Engine*)h)->cfg.max_len; }
int   laya_max_markers(void* h){ (void)h; return laya::Engine::KMAX; }
float laya_temperature(void* h,int qtype){ return ((laya::Engine*)h)->cfg.temperature[qtype]; }
void  laya_prof_report(void* h,int iters){ ((laya::Engine*)h)->prof.report(iters); }
void  laya_prof_reset(void* h){ ((laya::Engine*)h)->prof.acc.clear(); }
void  laya_tune_all(void* h){ ((laya::Engine*)h)->tune_all(); }
}
