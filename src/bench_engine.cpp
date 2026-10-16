#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>
#include "laya_api.h"
// Latency per call (p50) for B questions of S tokens each, 5 markers per question.
static double bench(void* e,int S,int B,int N){
  std::vector<int> lens(B,S),nm(B,5),ids((size_t)B*S),mk((size_t)B*5),qt(B,0);
  for(size_t i=0;i<ids.size();i++) ids[i]=100+(int)((i*7919)%40000);
  for(int b=0;b<B;b++) for(int k=0;k<5;k++) mk[b*5+k]=5+4*k;
  std::vector<float> lo((size_t)B*5),ac((size_t)B*8);
  for(int i=0;i<3;i++) laya_forward_batch(e,B,lens.data(),ids.data(),nm.data(),mk.data(),qt.data(),lo.data(),ac.data());
  std::vector<double> t(N);
  for(int i=0;i<N;i++){
    auto a=std::chrono::high_resolution_clock::now();
    laya_forward_batch(e,B,lens.data(),ids.data(),nm.data(),mk.data(),qt.data(),lo.data(),ac.data());
    t[i]=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-a).count();
  }
  std::sort(t.begin(),t.end()); return t[N/2];
}
int main(int argc,char**argv){
  void* e=laya_create(argc>1?argv[1]:"/weights/laya-en");
  int ML=laya_max_len(e);
  if(getenv("LAYA_TUNE_ALL")){ auto t=std::chrono::high_resolution_clock::now(); laya_tune_all(e);
    printf("tune_all: %.1f s\n",std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t).count()); }
  printf("\n-- single question --\n%-8s %10s %10s\n","tokens","p50_ms","q/s");
  for(int S:{64,128,256,384,512,768,1024}){ if(S>ML) break; double m=bench(e,S,1,30); printf("%-8d %10.2f %10.1f\n",S,m,1000/m); }
  printf("\n-- batched (questions per call, 256 tokens each) --\n%-8s %10s %10s %12s\n","batch","p50_ms","q/s","ms/question");
  for(int B:{1,2,4,8,16,32}){ double m=bench(e,256,B,15); printf("%-8d %10.2f %10.1f %12.2f\n",B,m,1000*B/m,m/B); }
  if(getenv("LAYA_LENS")){ printf("\n-- odd lengths --\n");
    for(int S=144;S<=320;S+=16){ double m=bench(e,S,1,20); printf("%-8d %10.2f\n",S,m); } }
  if(getenv("LAYA_PROFILE")){ int S=atoi(getenv("LAYA_PROFILE")); int PB=getenv("LAYA_PROFILE_B")?atoi(getenv("LAYA_PROFILE_B")):1; bench(e,S,PB,3); laya_prof_reset(e); bench(e,S,PB,5); laya_prof_report(e,8); }
  return 0;
}
