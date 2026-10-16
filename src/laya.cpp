#include "laya.hpp"
#include <cstring>
namespace laya {

Config Config::load(const std::string& encoder_cfg, const std::string& agent_cfg){
  Config c;
  { FILE* f=fopen(encoder_cfg.c_str(),"rb"); if(!f){fprintf(stderr,"cannot open %s\n",encoder_cfg.c_str());exit(1);} 
    std::string s; char buf[8192]; size_t n; while((n=fread(buf,1,sizeof buf,f))>0) s.append(buf,n); fclose(f);
    json j=json::parse(s);
    c.n_layers=j.value("num_hidden_layers",28); c.d=j.value("hidden_size",1024);
    c.n_heads=j.value("num_attention_heads",16); c.head_dim=c.d/c.n_heads;
    c.intermediate=j.value("intermediate_size",2624); c.vocab=j.value("vocab_size",50368);
    c.local_attention=j.value("local_attention",128);
    c.norm_eps=j.value("norm_eps", j.value("layer_norm_eps",1e-5));
    c.pad_id=j.value("pad_token_id",50283); c.cls_id=j.value("cls_token_id",50281);
    c.sep_id=j.value("sep_token_id",50282); c.mask_id=j.value("mask_token_id",50284);
    c.sans_pos = (j.value("position_embedding_type",std::string("absolute"))=="sans_pos");
    if(j.contains("rope_parameters")){
      auto& rp=j["rope_parameters"];
      if(rp.contains("full_attention")) c.rope_theta_global=rp["full_attention"].value("rope_theta",160000.0);
      if(rp.contains("sliding_attention")) c.rope_theta_local=rp["sliding_attention"].value("rope_theta",10000.0);
    }
    if(j.contains("layer_types")) for(auto& t:j["layer_types"])
      c.layer_types.push_back(t.get<std::string>()=="full_attention"?AttnKind::Global:AttnKind::Sliding);
    else { int every=j.value("global_attn_every_n_layers",3);
      for(int i=0;i<c.n_layers;i++) c.layer_types.push_back(i%every==0?AttnKind::Global:AttnKind::Sliding); }
  }
  { FILE* f=fopen(agent_cfg.c_str(),"rb"); if(f){ std::string s; char buf[8192]; size_t n;
      while((n=fread(buf,1,sizeof buf,f))>0) s.append(buf,n); fclose(f);
      json j=json::parse(s);
      c.max_len=j.value("max_len",512); c.head_max_len=j.value("head_max_len",192);
      c.head_layers=j.value("head_layers",2);
      if(j.contains("act_costs")) c.n_act=(int)j["act_costs"].size()+1;
      if(j.contains("temperature")) for(auto& t:j["temperature"]) c.temperature.push_back(t.get<float>());
      if(j.contains("temperature_by_options")) for(auto it=j["temperature_by_options"].begin();it!=j["temperature_by_options"].end();++it)
        c.temp_by_options[it.key()]=it.value().get<float>();
    } }
  if(c.temperature.empty()) c.temperature={1.f,1.f,1.f};
  c.head_ffn=4*c.d;
  return c;
}

void Config::dump() const{
  int ng=0; for(auto k:layer_types) if(k==AttnKind::Global) ng++;
  printf("[laya] layers=%d d=%d heads=%d head_dim=%d ffn=%d vocab=%d max_len=%d\n",
         n_layers,d,n_heads,head_dim,intermediate,vocab,max_len);
  printf("[laya] attn: %d global / %d sliding (window=%d) rope theta g=%.0f l=%.0f eps=%g sans_pos=%d\n",
         ng,n_layers-ng,local_attention,rope_theta_global,rope_theta_local,norm_eps,(int)sans_pos);
  printf("[laya] head: %d layers ffn=%d n_act=%d  temps=[%.4f %.4f %.4f]\n",
         head_layers,head_ffn,n_act,temperature[0],temperature[1],temperature[2]);
}

bool SafeTensors::open(const std::string& path){
  fd=::open(path.c_str(),O_RDONLY); if(fd<0){fprintf(stderr,"open %s failed\n",path.c_str());return false;}
  struct stat st; fstat(fd,&st); fsize=st.st_size;
  base=mmap(nullptr,fsize,PROT_READ,MAP_PRIVATE,fd,0);
  if(base==MAP_FAILED){fprintf(stderr,"mmap failed\n");return false;}
  uint64_t hlen; memcpy(&hlen,base,8); data_off=8+hlen;
  std::string hdr((const char*)base+8,hlen);
  json j=json::parse(hdr);
  for(auto it=j.begin();it!=j.end();++it){
    if(it.key()=="__metadata__") continue;
    Tensor t; t.dtype=it.value()["dtype"].get<std::string>();
    for(auto& s:it.value()["shape"]) t.shape.push_back(s.get<int64_t>());
    auto off=it.value()["data_offsets"];
    size_t a=off[0].get<size_t>(), b=off[1].get<size_t>();
    t.nbytes=b-a; t.data=(const char*)base+data_off+a;
    tensors[it.key()]=t;
  }
  return true;
}
const Tensor* SafeTensors::get(const std::string& n) const{ auto it=tensors.find(n); return it==tensors.end()?nullptr:&it->second; }
const Tensor& SafeTensors::must(const std::string& n) const{
  auto t=get(n); if(!t){fprintf(stderr,"[laya] missing tensor %s\n",n.c_str());exit(1);} return *t; }
SafeTensors::~SafeTensors(){ if(base&&base!=MAP_FAILED) munmap(base,fsize); if(fd>=0) ::close(fd); }

} // namespace laya
