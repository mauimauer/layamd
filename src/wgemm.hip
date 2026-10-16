// Custom WMMA GEMM for gfx1151 (RDNA3.5, wave32): C[M,N] = A[M,K] * Wt[N,K]^T.
//
// Both operands are K-contiguous (activations row-major, weights in their original PyTorch
// [out,in] layout), so every WMMA fragment is a contiguous 32-byte row slice.
// 8 waves per workgroup, BM x BN tile, 32-wide K steps staged through double-buffered LDS
// (global loads for step s+1 are in flight while step s computes). Optional split-K over
// grid.z writes fp16 partials [z][M][N] for the existing reduce / residual+LN kernels.
//
// GEGLU=true fuses ModernBERT's GeGLU into the epilogue: the tile's W rows are BN/2 "input"
// rows [c0, c0+BN/2) and the matching BN/2 "gate" rows [I+c0, ...), and the kernel writes
// out[m][c] = gelu(in) * gate directly -- the [M, 2I] intermediate never touches memory.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include "wgemm.h"

typedef _Float16 wh16 __attribute__((ext_vector_type(16)));
typedef _Float16 wh8 __attribute__((ext_vector_type(8)));
typedef float wf8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float wg_gelu(float x){ return 0.5f*x*(1.f+erff(x*0.70710678118654752f)); }

template<int BM,int BN,bool GEGLU,int KS=32,int NW=8,int FM=2>
__global__ __launch_bounds__(NW*32) void k_wgemm(const __half* __restrict__ A, const __half* __restrict__ Wt,
    __half* __restrict__ C, int M, int N, int K, int ldc, int splits, size_t zstride, int I){
  constexpr int LDK=KS+8;                           // LDS row = KS+8 halfs: 16B-aligned, bank-spread
  constexpr int NT=NW*32, WR=BM/(16*FM), WC=NW/WR, WN=BN/WC, NF=WN/16;
  static_assert(WR*WC==NW && NF>=1, "tile");
  static_assert(!GEGLU || NF%2==0, "geglu needs an even number of column fragments per wave");
  __shared__ __attribute__((aligned(16))) _Float16 As[2][BM][LDK];
  __shared__ __attribute__((aligned(16))) _Float16 Ws[2][BN][LDK];
  const int t=threadIdx.x, wave=t>>5, lane=t&31, r16=lane&15, hl=lane>>4;
  const int wr=wave/WC, wc=wave%WC;
  const int m0=blockIdx.y*BM, bn=blockIdx.x, z=blockIdx.z;
  const int nsteps=K/KS, s0=(int)((long)nsteps*z/splits), s1=(int)((long)nsteps*(z+1)/splits);

  // W tile row r -> global weight row
  auto wrow=[&](int r)->int{
    if(!GEGLU) return bn*BN+r;
    return r<BN/2 ? bn*(BN/2)+r : I+bn*(BN/2)+(r-BN/2);
  };
  // column fragment g of this wave -> row inside the W tile
  auto fragrow=[&](int g)->int{
    if(!GEGLU) return wc*WN+g*16;
    return g<NF/2 ? wc*(WN/2)+g*16 : BN/2+wc*(WN/2)+(g-NF/2)*16;
  };

  constexpr int ACH=BM*KS/8, WCH=BN*KS/8;           // 16-byte chunks per tile
  constexpr int AP=(ACH+NT-1)/NT, WP=(WCH+NT-1)/NT;
  wh8 ra[AP], rw[WP];
  auto gload=[&](int s){
    const int k0=s*KS;
    #pragma unroll
    for(int p=0;p<AP;p++){ int c=t+p*NT; if(c<ACH){ int r=c/(KS/8), q=c%(KS/8);
      ra[p]=*(const wh8*)((const _Float16*)A+(size_t)(m0+r)*K+k0+q*8); } }
    #pragma unroll
    for(int p=0;p<WP;p++){ int c=t+p*NT; if(c<WCH){ int r=c/(KS/8), q=c%(KS/8);
      rw[p]=*(const wh8*)((const _Float16*)Wt+(size_t)wrow(r)*K+k0+q*8); } }
  };
  auto lstore=[&](int buf){
    #pragma unroll
    for(int p=0;p<AP;p++){ int c=t+p*NT; if(c<ACH){ int r=c/(KS/8), q=c%(KS/8); *(wh8*)&As[buf][r][q*8]=ra[p]; } }
    #pragma unroll
    for(int p=0;p<WP;p++){ int c=t+p*NT; if(c<WCH){ int r=c/(KS/8), q=c%(KS/8); *(wh8*)&Ws[buf][r][q*8]=rw[p]; } }
  };
  auto frag=[&](const _Float16* row)->wh16{       // 16 contiguous halfs from an LDS row (two 16B reads)
    wh8 lo=*(const wh8*)row, hi=*(const wh8*)(row+8);
    return (wh16){lo[0],lo[1],lo[2],lo[3],lo[4],lo[5],lo[6],lo[7],hi[0],hi[1],hi[2],hi[3],hi[4],hi[5],hi[6],hi[7]};
  };

  wf8 acc[FM][NF];
  #pragma unroll
  for(int f=0;f<FM;f++)
    #pragma unroll
    for(int g=0;g<NF;g++) acc[f][g]=(wf8){0,0,0,0,0,0,0,0};

  if(s0<s1){ gload(s0); lstore(0); }
  __syncthreads();
  int buf=0;
  for(int s=s0;s<s1;s++){
    if(s+1<s1) gload(s+1);                          // next tile's global loads overlap this compute
    #pragma unroll
    for(int kk=0;kk<KS;kk+=16){
      wh16 af[FM], bf[NF];
      #pragma unroll
      for(int f=0;f<FM;f++) af[f]=frag(&As[buf][wr*16*FM+f*16+r16][kk]);
      #pragma unroll
      for(int g=0;g<NF;g++) bf[g]=frag(&Ws[buf][fragrow(g)+r16][kk]);
      #pragma unroll
      for(int f=0;f<FM;f++)
        #pragma unroll
        for(int g=0;g<NF;g++) acc[f][g]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[f],bf[g],acc[f][g]);
    }
    if(s+1<s1){ lstore(buf^1); }
    __syncthreads();
    buf^=1;
  }

  // epilogue. C/D fragment: element i of lane l is row 2i + l/16, col l%16.
  __half* Cz=C+(size_t)z*zstride;
  #pragma unroll
  for(int f=0;f<FM;f++){
    #pragma unroll
    for(int i=0;i<8;i++){
      const int m=m0+wr*16*FM+f*16+2*i+hl;
      if(!GEGLU){
        #pragma unroll
        for(int g=0;g<NF;g++) Cz[(size_t)m*ldc+bn*BN+wc*WN+g*16+r16]=__float2half(acc[f][g][i]);
      } else {
        #pragma unroll
        for(int g=0;g<NF/2;g++){
          const int col=bn*(BN/2)+wc*(WN/2)+g*16+r16;
          Cz[(size_t)m*ldc+col]=__float2half(wg_gelu(acc[f][g][i])*acc[f][g+NF/2][i]);
        }
      }
    }
  }
}

template<int BM,int BN,bool G,int KS=32,int NW=8,int FM=2>
static void launch_t(const __half* A,const __half* Wt,__half* C,int M,int N,int K,int splits,int I,hipStream_t st){
  const int ncols = G ? I : N;                       // output columns
  dim3 grid(G ? I/(BN/2) : N/BN, M/BM, splits);
  hipLaunchKernelGGL((k_wgemm<BM,BN,G,KS,NW,FM>),grid,dim3(NW*32),0,st,A,Wt,C,M,N,K,ncols,splits,(size_t)M*ncols,I);
}

// cfg: tile BMxBN, K step, waves, row-fragments per wave (wave tile = 16*FM x BN/(NW/(BM/(16*FM))))
struct WCfg{ int BM,BN,KS; };
static const WCfg WCFG[]={{128,128,32},{64,128,32},{128,64,64},{64,128,64},{128,128,32},{128,64,32},{64,64,64}};
int wgemm_ncfg(){ return (int)(sizeof(WCFG)/sizeof(WCFG[0])); }

bool wgemm_supported(int cfg,int M,int N,int K,bool geglu,int I){
  const WCfg c=WCFG[cfg];
  if(M%c.BM || K%c.KS) return false;
  if(geglu) return I%(c.BN/2)==0;
  return N%c.BN==0;
}

#define WL(BM,BN,KS,NW,FM) { if(geglu) launch_t<BM,BN,true,KS,NW,FM>(A,Wt,C,M,N,K,1,I,st); else launch_t<BM,BN,false,KS,NW,FM>(A,Wt,C,M,N,K,splits,I,st); } break;
void launch_wgemm(int cfg,const __half* A,const __half* Wt,__half* C,int M,int N,int K,int splits,bool geglu,int I,hipStream_t st){
  switch(cfg){
    case 0: WL(128,128,32,8,2)
    case 1: WL(64,128,32,8,2)
    case 2: WL(128,64,64,8,2)
    case 3: WL(64,128,64,8,2)
    case 4: WL(128,128,32,4,4)     // 4 waves, 64x64 per wave
    case 5: WL(128,64,32,4,2)      // 4 waves, 32x64 per wave
    case 6: WL(64,64,64,4,2)       // 4 waves, 32x32 per wave
  }
}
