// SPDX-License-Identifier: Apache-2.0
//
// humming_launch.hpp: inkling-cmp-engine S4 prefill MoE bench, the Humming baseline arm. Copied from
// probes/humming/h2_harness.cpp (our H2 harness, which carries the attribution below) and cut down to what the
// prefill bench needs: minimal JSON + safetensors reader, the cubin reader, the kernel metadata, the module load and
// the 18-argument launch.
//
// Attribution (Apache-2.0, section 4): the cubin reader (CubinReader), the launch metadata field set and the launch
// sequence (argument array, CUlaunchConfig, cuFuncSetAttribute, cuLaunchKernelEx) are adapted from Humming 0.1.20
// (https://github.com/vllm-project/humming, commit 0832818a8), files csrc/launcher/elf.h, csrc/launcher/launcher.cpp and
// csrc/launcher/utils.h, Copyright the Humming contributors, licensed under the Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0). Changes from the original: torch types and the dispatcher removed;
// tensors replaced by raw CUdeviceptr; TMA, cluster and PDL paths replaced by assertions that they are off (sm_80
// config B cubins); here also: the H1-manifest checks removed (the prefill cubins are taken from config B's JIT cache
// by md5), safetensors reader returns dtype/shape. The cubins themselves are Humming object code (unmodified) and stay
// outside this repo.
#pragma once
#include <cuda.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace hum {
// ------------------------------------------------------------------------------------------------ errors, logging
[[noreturn]] static void die(const std::string &msg) {
  std::fprintf(stderr, "FATAL: %s\n", msg.c_str());
  std::fflush(stderr);
  std::exit(2);
}
#define REQUIRE(cond, msg)                                                                     \
  do {                                                                                         \
    if (!(cond)) ::hum::die(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + (msg));    \
  } while (0)

// As H/csrc/launcher/utils.h:19-27 (check_curesult).
static void check_cu(CUresult r, const char *what) {
  if (r == CUDA_SUCCESS) return;
  const char *n = nullptr, *s = nullptr;
  cuGetErrorName(r, &n);
  cuGetErrorString(r, &s);
  die(std::string(what) + " failed with error: " + (n ? n : "?") + " (" + (s ? s : "?") + ")");
}
#define CU(call) ::hum::check_cu((call), #call)
// ------------------------------------------------------------------------------------------------ minimal JSON
struct Json {
  enum T { Null, Bool, Num, Str, Arr, Obj } t = Null;
  bool b = false;
  double n = 0;
  std::string s;
  std::vector<Json> a;
  std::vector<std::pair<std::string, Json>> o;
  const Json *find(const std::string &k) const {
    for (auto &kv : o)
      if (kv.first == k) return &kv.second;
    return nullptr;
  }
  const Json &operator[](const std::string &k) const {
    auto *p = find(k);
    REQUIRE(p, "json key missing: " + k);
    return *p;
  }
  const Json &operator[](size_t i) const { return a.at(i); }
  bool null() const { return t == Null; }
  int64_t i() const { return static_cast<int64_t>(std::llround(n)); }
  uint64_t u() const { return static_cast<uint64_t>(std::llround(n)); }
  const std::string &str() const {
    REQUIRE(t == Str, "json: not a string");
    return s;
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string &text) : p_(text.data()), e_(text.data() + text.size()) {}
  Json parse() {
    Json v = value();
    ws();
    return v;
  }

 private:
  const char *p_, *e_;
  void ws() {
    while (p_ < e_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_;
  }
  [[noreturn]] void err(const char *m) { die(std::string("json parse: ") + m); }
  Json value() {
    ws();
    if (p_ >= e_) err("eof");
    Json v;
    char c = *p_;
    if (c == '{') {
      v.t = Json::Obj;
      ++p_;
      ws();
      if (*p_ == '}') { ++p_; return v; }
      for (;;) {
        ws();
        std::string k = string();
        ws();
        if (*p_++ != ':') err("expected :");
        v.o.emplace_back(std::move(k), value());
        ws();
        if (*p_ == ',') { ++p_; continue; }
        if (*p_ == '}') { ++p_; return v; }
        err("expected , or }");
      }
    } else if (c == '[') {
      v.t = Json::Arr;
      ++p_;
      ws();
      if (*p_ == ']') { ++p_; return v; }
      for (;;) {
        v.a.push_back(value());
        ws();
        if (*p_ == ',') { ++p_; continue; }
        if (*p_ == ']') { ++p_; return v; }
        err("expected , or ]");
      }
    } else if (c == '"') {
      v.t = Json::Str;
      v.s = string();
    } else if (!std::strncmp(p_, "true", 4)) {
      v.t = Json::Bool; v.b = true; p_ += 4;
    } else if (!std::strncmp(p_, "false", 5)) {
      v.t = Json::Bool; p_ += 5;
    } else if (!std::strncmp(p_, "null", 4)) {
      p_ += 4;
    } else if (!std::strncmp(p_, "NaN", 3)) {
      v.t = Json::Num; v.n = NAN; p_ += 3;
    } else {
      char *end = nullptr;
      v.t = Json::Num;
      v.n = std::strtod(p_, &end);
      if (end == p_) err("bad value");
      p_ = end;
    }
    return v;
  }
  std::string string() {
    if (*p_ != '"') err("expected string");
    ++p_;
    std::string out;
    while (p_ < e_ && *p_ != '"') {
      if (*p_ == '\\') {
        ++p_;
        switch (*p_) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'u': out += '?'; p_ += 4; break;
          default: out += *p_;
        }
        ++p_;
      } else {
        out += *p_++;
      }
    }
    ++p_;
    return out;
  }
};

static std::string read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE(f, "cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
// ------------------------------------------------------------------------------------------------ safetensors
class SafeTensors {
 public:
  struct Info { std::string dtype; std::vector<int64_t> shape; uint64_t a = 0, b = 0; };
  explicit SafeTensors(const std::string &path) : path_(path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f, "cannot open " + path);
    uint64_t hlen = 0;
    f.read(reinterpret_cast<char *>(&hlen), 8);
    std::string hdr(hlen, '\0');
    f.read(hdr.data(), static_cast<std::streamsize>(hlen));
    data_start_ = 8 + hlen;
    Json j = JsonParser(hdr).parse();
    for (auto &kv : j.o) {
      if (kv.first == "__metadata__") continue;
      Info in;
      in.dtype = kv.second["dtype"].str();
      for (auto &d : kv.second["shape"].a) in.shape.push_back(d.i());
      in.a = kv.second["data_offsets"][0].u();
      in.b = kv.second["data_offsets"][1].u();
      off_[kv.first] = in;
    }
  }
  bool has(const std::string &n) const { return off_.count(n) != 0; }
  const Info &info(const std::string &n) const {
    auto it = off_.find(n);
    REQUIRE(it != off_.end(), "tensor missing in " + path_ + ": " + n);
    return it->second;
  }
  size_t size(const std::string &n) const { return info(n).b - info(n).a; }
  std::vector<uint8_t> read(const std::string &n) const {
    const Info &in = info(n);
    std::vector<uint8_t> v(in.b - in.a);
    std::ifstream f(path_, std::ios::binary);
    f.seekg(static_cast<std::streamoff>(data_start_ + in.a));
    f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(v.size()));
    REQUIRE(f, "short read " + n);
    return v;
  }

 private:
  std::string path_;
  uint64_t data_start_ = 0;
  std::map<std::string, Info> off_;
};

// ------------------------------------------------------------------------------------------------ cubin reader
// Port of H/csrc/launcher/elf.h:59-133 (CubinReader). Kernel entries are STT_FUNC symbols with st_other & 0x10
// (CUDA; 0x20 on PPU, elf.h:81-82); a constant's value is at sh_offset(section) + st_value; SHT_NOBITS -> 0.
#pragma pack(push, 1)
struct Elf64Ehdr {
  unsigned char e_ident[16];
  uint16_t e_type, e_machine;
  uint32_t e_version;
  uint64_t e_entry, e_phoff, e_shoff;
  uint32_t e_flags;
  uint16_t e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx;
};
struct Elf64Shdr {
  uint32_t sh_name, sh_type;
  uint64_t sh_flags, sh_addr, sh_offset, sh_size;
  uint32_t sh_link, sh_info;
  uint64_t sh_addralign, sh_entsize;
};
struct Elf64Sym {
  uint32_t st_name;
  unsigned char st_info, st_other;
  uint16_t st_shndx;
  uint64_t st_value, st_size;
};
#pragma pack(pop)

class CubinReader {
 public:
  explicit CubinReader(const std::vector<char> &data) : data_(data) {
    REQUIRE(data_.size() >= sizeof(Elf64Ehdr), "cubin too small");
    auto *eh = reinterpret_cast<const Elf64Ehdr *>(data_.data());
    auto *sh = reinterpret_cast<const Elf64Shdr *>(data_.data() + eh->e_shoff);
    for (int i = 0; i < eh->e_shnum; ++i)
      if (sh[i].sh_type == 2) read_symtab(sh, eh->e_shnum, i, eh->e_machine == 0x10E ? 0x20 : 0x10);
  }
  uint32_t u32(const std::string &name) const {  // elf.h:87-97 getUint32
    auto it = off_.find(name);
    REQUIRE(it != off_.end(), "cubin symbol not found: " + name);
    if (it->second == 0) return 0;
    uint32_t v = 0;
    std::memcpy(&v, data_.data() + it->second, 4);
    return v;
  }
  bool b(const std::string &name) const { return u32(name) > 0; }  // elf.h:99-101
  const std::vector<std::string> &kernels() const { return kernels_; }

 private:
  const std::vector<char> &data_;
  std::map<std::string, uint64_t> off_;
  std::vector<std::string> kernels_;
  void read_symtab(const Elf64Shdr *sh, int nsh, int idx, unsigned char kflag) {  // elf.h:107-132
    const Elf64Shdr &st = sh[idx];
    const Elf64Shdr &strtab = sh[st.sh_link];
    const char *names = data_.data() + strtab.sh_offset;
    auto *syms = reinterpret_cast<const Elf64Sym *>(data_.data() + st.sh_offset);
    int count = static_cast<int>(st.sh_size / st.sh_entsize);
    for (int i = 0; i < count; ++i) {
      const Elf64Sym &s = syms[i];
      if (s.st_name == 0 || s.st_name >= strtab.sh_size) continue;
      if (s.st_shndx >= nsh) continue;
      std::string nm = names + s.st_name;
      if ((s.st_info & 0xF) == 2 && (s.st_other & kflag) != 0) kernels_.push_back(nm);
      const auto &tsh = sh[s.st_shndx];
      off_[nm] = tsh.sh_type == 8 /*SHT_NOBITS*/ ? 0 : tsh.sh_offset + s.st_value;
    }
  }
};

// The subset of H/csrc/launcher/utils.h:125-185 (KernelData) that launch_kernel_impl uses on this path, read with the
// same symbol names as launcher.cpp:282-341.
struct KernelMeta {
  uint32_t smem_size, num_threads, a_dtype_id, b_dtype_id, c_dtype_id, bs_dtype_id, problem_shape_n, problem_shape_k,
      pad_shape_n, pad_shape_k, block_shape_m, block_shape_n, block_shape_k, num_experts, num_ctas_per_sm,
      umma_cta_group_size, multi_cast_size_a, multi_cast_size_b, gemm_type_id, mma_type_id, num_stages;
  bool use_stream_k, use_pdl, use_batch_invariant, is_tensor_weight_scale_2, any_tma;
};
static KernelMeta read_meta(const CubinReader &r) {
  KernelMeta m{};
  m.smem_size = r.u32("SMEM_SIZE");
  m.num_threads = r.u32("NUM_THREADS");
  m.a_dtype_id = r.u32("A_DTYPE_ID");
  m.b_dtype_id = r.u32("B_DTYPE_ID");
  m.c_dtype_id = r.u32("C_DTYPE_ID");
  m.bs_dtype_id = r.u32("BS_DTYPE_ID");
  m.problem_shape_n = r.u32("PROBLEM_SHAPE_N");
  m.problem_shape_k = r.u32("PROBLEM_SHAPE_K");
  m.pad_shape_n = r.u32("PAD_SHAPE_N");
  m.pad_shape_k = r.u32("PAD_SHAPE_K");
  m.block_shape_m = r.u32("BLOCK_SHAPE_M");
  m.block_shape_n = r.u32("BLOCK_SHAPE_N");
  m.block_shape_k = r.u32("BLOCK_SHAPE_K");
  m.num_experts = r.u32("NUM_EXPERTS");
  m.num_ctas_per_sm = r.u32("NUM_CTAS_PER_SM");
  m.umma_cta_group_size = r.u32("UMMA_CTA_GROUP_SIZE");
  m.multi_cast_size_a = r.u32("MULTI_CAST_SIZE_A");
  m.multi_cast_size_b = r.u32("MULTI_CAST_SIZE_B");
  m.gemm_type_id = r.u32("GEMM_TYPE_ID");
  m.mma_type_id = r.u32("MMA_TYPE_ID");
  m.num_stages = r.u32("NUM_STAGES");
  m.use_stream_k = r.b("USE_STREAM_K");
  m.use_pdl = r.b("USE_PDL");
  m.use_batch_invariant = r.b("USE_BATCH_INVARIANT");
  m.is_tensor_weight_scale_2 = r.b("IS_TENSOR_WEIGHT_SCALE_2");
  m.any_tma = false;
  for (const char *n : {"USE_TMA_A", "USE_TMA_AS", "USE_TMA_AS2", "USE_TMA_B", "USE_TMA_C", "USE_TMA_BS",
                        "USE_TMA_BS2", "USE_TMA_BZP", "USE_TMA_BIAS"})
    m.any_tma |= r.b(n);
  return m;
}

struct Kernel {
  std::string md5, name;
  std::vector<char> cubin;
  KernelMeta m{};
  CUmodule mod = nullptr;
  CUfunction fn = nullptr;
  int regs = 0, local_bytes = 0;
};

// launcher.cpp:265-281 (register_kernel: exactly one symbol containing "humming") + :55-83 (cuModuleLoadData,
// cuModuleGetFunction) + :258-259 (cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES = SMEM_SIZE).
static std::unique_ptr<Kernel> load_kernel(const std::string &md5, const std::string &path) {
  auto k = std::make_unique<Kernel>();
  k->md5 = md5;
  std::string bytes = read_file(path);
  k->cubin.assign(bytes.begin(), bytes.end());
  CubinReader r(k->cubin);
  for (auto &n : r.kernels()) {
    if (n.find("humming") == std::string::npos) continue;
    REQUIRE(k->name.empty(), "multiple humming kernels in " + path);
    k->name = n;
  }
  REQUIRE(!k->name.empty(), "no humming kernel in " + path);
  k->m = read_meta(r);
  REQUIRE(!k->m.any_tma, "cubin uses TMA: not supported here " + md5);
  REQUIRE(!k->m.use_pdl, "cubin uses PDL " + md5);
  REQUIRE(k->m.multi_cast_size_a * k->m.multi_cast_size_b * k->m.umma_cta_group_size == 1, "cluster > 1 " + md5);
  REQUIRE(!k->m.use_stream_k, "cubin uses Stream-K " + md5);
  REQUIRE(k->m.gemm_type_id == 1, "not an indexed (MoE) GEMM " + md5);
  REQUIRE(k->m.mma_type_id == 0, "not MmaType::MMA " + md5);
  CU(cuModuleLoadData(&k->mod, k->cubin.data()));
  CU(cuModuleGetFunction(&k->fn, k->mod, k->name.c_str()));
  CU(cuFuncSetAttribute(k->fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, static_cast<int>(k->m.smem_size)));
  CU(cuFuncGetAttribute(&k->regs, CU_FUNC_ATTRIBUTE_NUM_REGS, k->fn));
  CU(cuFuncGetAttribute(&k->local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, k->fn));
  return k;
}

// One Humming GEMM launch with the 18-argument ABI of H/include/humming/kernel/humming.cuh:32-50.
// Argument order and by-pointer convention: launcher.cpp:204-222. With every USE_TMA_* = 0 (asserted at load) each of
// the first nine arguments is `void *const` (humming.cuh:17-21 KernelTensorParamType<false>), i.e. 8 bytes.
// cuobjdump -elf of every config B GEMM cubin: EIATTR_KPARAM_INFO ordinals 0..17, offsets 0x0..0x80, sizes 8 x 15,
// 4, 4, 1; EIATTR_CBANK_PARAM_SIZE 0x81 (checked on the 52 H0 GEMM cubins).
struct Launch {
  const Kernel *k = nullptr;
  uint32_t grid_x = 0;                     // NUM_CTAS_PER_SM * num_sms (launcher.cpp:150, :225, :235)
  CUdeviceptr A = 0;                       // arg 0  a.data_ptr()          (x [T,2304] bf16 | act [8T,1152] bf16)
  CUdeviceptr B = 0;                       // arg 1  b.data_ptr()          repacked weight int32 (transform.py)
  CUdeviceptr C = 0;                       // arg 2  c.data_ptr()          output bf16 [out_rows, 2304]
  CUdeviceptr AS = 0;                      // arg 3  as_  -> nullptr        (bf16 activations: input_scale None)
  CUdeviceptr AS2 = 0;                     // arg 4  as2  -> nullptr
  CUdeviceptr BS = 0;                      // arg 5  bs.data_ptr()         weight_scale (e4m3 [E,K/16,N] | e8m0 [E,K/32,N])
  CUdeviceptr BZP = 0;                     // arg 6  bzp  -> nullptr        (no zero point)
  CUdeviceptr Bias = 0;                    // arg 7  bias -> nullptr
  CUdeviceptr BS2 = 0;                     // arg 8  bs2.data_ptr()        weight_scale_2 fp32 [E] (NVFP4) | as H1 saw it
  CUdeviceptr sorted_ids = 0;              // arg 9  moe_align_block_size sorted_token_ids (int32, read as uint32)
  CUdeviceptr expert_ids = 0;              // arg 10 per-m-block expert id
  CUdeviceptr num_tokens_padded = 0;       // arg 11 [1], read on the device (scheduler.cuh:157-159)
  CUdeviceptr expert_layout = 0;           // arg 12 nullptr (indexed GEMM; launcher.cpp:180)
  CUdeviceptr tensor_map_buffer = 0;       // arg 13 torch_empty({0}) -> nullptr here (tensor.h:19-29; unused, no TMA C)
  CUdeviceptr locks = 0;                   // arg 14 HummingExperts.locks (2048 int32 zeros, moe_sm90.py:244)
  uint32_t shape_m = 0;                    // arg 15 a.size(0) (launcher.cpp:142,199): T for w13, 8T for w2
  uint32_t top_k = 0;                      // arg 16 8 for w13, 1 for w2 (moe_sm90.py:310,315)
  bool use_int64_expert_layout = false;    // arg 17 false (no expert_layout; launcher.cpp:194-197)
};

static void launch(const Launch &L, CUstream stream) {
  void *a_ptr = reinterpret_cast<void *>(L.A), *b_ptr = reinterpret_cast<void *>(L.B),
       *c_ptr = reinterpret_cast<void *>(L.C), *as_ptr = reinterpret_cast<void *>(L.AS),
       *as2_ptr = reinterpret_cast<void *>(L.AS2), *bs_ptr = reinterpret_cast<void *>(L.BS),
       *bzp_ptr = reinterpret_cast<void *>(L.BZP), *bias_ptr = reinterpret_cast<void *>(L.Bias),
       *bs2_ptr = reinterpret_cast<void *>(L.BS2), *sorted_ids_ptr = reinterpret_cast<void *>(L.sorted_ids),
       *expert_ids_ptr = reinterpret_cast<void *>(L.expert_ids),
       *num_tokens_padded_ptr = reinterpret_cast<void *>(L.num_tokens_padded),
       *expert_layout_ptr = reinterpret_cast<void *>(L.expert_layout),
       *tensor_map_buffer_ptr = reinterpret_cast<void *>(L.tensor_map_buffer),
       *locks_ptr = reinterpret_cast<void *>(L.locks);
  uint32_t shape_m_u32 = L.shape_m, top_k_u32 = L.top_k;
  bool use_int64_expert_layout = L.use_int64_expert_layout;
  void *kernel_args[] = {&a_ptr, &b_ptr, &c_ptr, &as_ptr, &as2_ptr, &bs_ptr, &bzp_ptr, &bias_ptr, &bs2_ptr,
                         &sorted_ids_ptr, &expert_ids_ptr, &num_tokens_padded_ptr, &expert_layout_ptr,
                         &tensor_map_buffer_ptr, &locks_ptr, &shape_m_u32, &top_k_u32, &use_int64_expert_layout};
  static_assert(sizeof(kernel_args) / sizeof(void *) == 18, "18-argument ABI");
  CUlaunchConfig config = {};              // launcher.cpp:224-256 (no attributes: cluster 1, no PDL)
  config.gridDimX = L.grid_x;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = L.k->m.num_threads;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = L.k->m.smem_size;
  config.hStream = stream;
  config.attrs = nullptr;
  config.numAttrs = 0;
  CU(cuLaunchKernelEx(&config, L.k->fn, kernel_args, nullptr));
}

}  // namespace hum
