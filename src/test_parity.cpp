// Validate the HIP engine against golden tensors from the PyTorch reference.
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include "third_party/json.hpp"
#include "laya_api.h"
using json = nlohmann::json;

static std::vector<int> readi(const std::string& p){
  std::ifstream f(p,std::ios::binary); if(!f){fprintf(stderr,"missing %s\n",p.c_str());exit(1);}
  f.seekg(0,std::ios::end); size_t n=f.tellg()/4; f.seekg(0);
  std::vector<int> v(n); f.read((char*)v.data(),n*4); return v; }
static std::vector<float> readf(const std::string& p){
  std::ifstream f(p,std::ios::binary); if(!f){fprintf(stderr,"missing %s\n",p.c_str());exit(1);}
  f.seekg(0,std::ios::end); size_t n=f.tellg()/4; f.seekg(0);
  std::vector<float> v(n); f.read((char*)v.data(),n*4); return v; }

int main(int argc,char** argv){
  std::string model=argc>1?argv[1]:"/weights/laya-en";
  std::string gold =argc>2?argv[2]:"/golden";
  std::ifstream mf(gold+"/manifest.json"); json M; mf>>M;
  void* e=laya_create(model.c_str());
  int nact=laya_n_act(e);
  printf("\n%-16s %4s %5s %4s   %-11s %-11s %-11s  %s\n",
         "case","row","len","K","max|dlogit|","max|dprob|","rel|dact|","verdict");
  double worst_p=0; int fails=0, total=0;
  // pass 0 runs eagerly and captures a graph per shape; pass 1 replays the graphs
  for(int pass=0;pass<2;pass++) for(auto& c:M["cases"]){
    std::string name=c["name"], dir=gold+"/"+name;
    auto ids=readi(dir+"/input_ids.bin");
    auto att=readi(dir+"/attention_mask.bin");
    auto mpos=readi(dir+"/marker_pos.bin");
    auto mmask=readi(dir+"/marker_mask.bin");
    auto qt=readi(dir+"/qtype.bin");
    auto glog=readf(dir+"/logits.bin");
    auto gact=readf(dir+"/act_logits.bin");
    int n=c["shapes"]["input_ids"][0], L=c["shapes"]["input_ids"][1];
    int kmax=c["shapes"]["marker_pos"][1];
    // whole case as ONE batched call (multi_q exercises B=3 with unequal lengths)
    std::vector<int> lens(n),nm(n),cat_ids,cat_mk,qts(n);
    for(int r=0;r<n;r++){
      int vlen=0; for(int i=0;i<L;i++) vlen+=att[r*L+i];
      int K=0; for(int i=0;i<kmax;i++) K+=mmask[r*kmax+i];
      lens[r]=vlen; nm[r]=K; qts[r]=qt[r];
      cat_ids.insert(cat_ids.end(),&ids[(size_t)r*L],&ids[(size_t)r*L]+vlen);
      cat_mk.insert(cat_mk.end(),&mpos[(size_t)r*kmax],&mpos[(size_t)r*kmax]+K);
    }
    std::vector<float> allo(cat_mk.size()),allac((size_t)n*nact);
    if(laya_forward_batch(e,n,lens.data(),cat_ids.data(),nm.data(),cat_mk.data(),qts.data(),allo.data(),allac.data())){
      printf("%s: forward_batch rejected input\n",name.c_str()); return 2; }
    size_t mo=0;
    for(int r=0;r<n;r++){
      int vlen=lens[r], K=nm[r];
      float* lo=&allo[mo]; float* ac=&allac[(size_t)r*nact]; mo+=K;
      double dl=0,dp=0,da=0;
      auto softmax=[&](const float* z,int k,std::vector<double>& p){
        double mx=-1e30; for(int i=0;i<k;i++) mx=std::max(mx,(double)z[i]);
        double s=0; p.resize(k); for(int i=0;i<k;i++){p[i]=exp(z[i]-mx);s+=p[i];}
        for(int i=0;i<k;i++) p[i]/=s; };
      std::vector<double> pe,pg;
      softmax(lo,K,pe); softmax(&glog[(size_t)r*kmax],K,pg);
      for(int i=0;i<K;i++){ dl=std::max(dl,(double)fabs(lo[i]-glog[(size_t)r*kmax+i])); dp=std::max(dp,fabs(pe[i]-pg[i])); }
      // act head is saturated (|logit| ~ 1e3), so compare relative error
      for(int i=0;i<nact;i++){ double g=gact[(size_t)r*nact+i]; da=std::max(da,fabs(ac[i]-g)/std::max(1.0,fabs(g))); }
      bool ok = dp<0.01 && da<0.05;
      worst_p=std::max(worst_p,dp); total++; if(!ok) fails++;
      printf("%-16s %4d %5d %4d   %11.5f %11.6f %11.5f  %s\n",
             (name+(pass?" [graph]":"")).c_str(),r,vlen,K,dl,dp,da, ok?"PASS":"** FAIL **");
    }
  }
  printf("\n%d/%d rows pass; worst probability delta = %.6f\n", total-fails, total, worst_p);
  return fails?1:0;
}
