// Laya inference engine — config + weight loading. Config-driven so the same binary
// serves ModernBERT-large (en / typed-decisions) and mmBERT-base (multilingual).
#pragma once
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "third_party/json.hpp"
using json = nlohmann::json;

namespace laya {

enum class AttnKind { Global, Sliding };

struct Config {
  int n_layers=28, d=1024, n_heads=16, head_dim=64, intermediate=2624;
  int vocab=50368, max_len=512, head_max_len=192, head_layers=2, n_act=2;
  int local_attention=128;              // full window width; half each side
  float norm_eps=1e-5f;
  float rope_theta_global=160000.f, rope_theta_local=10000.f;
  int pad_id=50283, cls_id=50281, sep_id=50282, mask_id=50284;
  std::vector<AttnKind> layer_types;
  std::vector<float> temperature;                    // per qtype
  std::map<std::string,float> temp_by_options;
  int head_ffn=0;                                    // 4*d
  bool sans_pos=false;                               // multilingual mmBERT variant

  static Config load(const std::string& encoder_cfg, const std::string& agent_cfg);
  void dump() const;
};

struct Tensor { const void* data=nullptr; std::vector<int64_t> shape; std::string dtype; size_t nbytes=0; };

// mmap'd safetensors file; tensors are views into the mapping (no copy).
struct SafeTensors {
  int fd=-1; void* base=nullptr; size_t fsize=0; size_t data_off=0;
  std::map<std::string,Tensor> tensors;
  bool open(const std::string& path);
  const Tensor* get(const std::string& name) const;
  const Tensor& must(const std::string& name) const;
  ~SafeTensors();
};

} // namespace laya
