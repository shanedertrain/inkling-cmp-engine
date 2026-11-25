// Kernel pack: layout of every launch-argument struct, from the C++ definitions themselves (own code, package R2,
// docs/RUST-PLAN.md 3.2). Host program, compiled by nvcc with the pack's headers and flags (build.sh) and run on the
// build host (no GPU). Writes layout.json to stdout:
//   {"format": "ink-kpack-layout/1", "structs": [{"name", "size", "align",
//     "fields": [{"name", "type", "offset", "size", "align"}, ...]}, ...]}
// in structs.def order; `type` is the demangled typeid name (e.g. "float const*", "float const* [4]", "pfa::Seg const*").
// Fails (exit 1) when a struct's listed fields plus alignment padding do not tile sizeof exactly (a layout the Rust
// mirror could not reproduce), or when the struct has members structs.def does not list. The tiling check alone misses
// a 4-byte member hidden in padding (R3: projs::Job::cta0 sat in Job's tail padding, 44..48); the member count is
// therefore checked too, by aggregate initialisation (count_inits below: the most initialisers `S{...}` accepts,
// one per member, one per array element by brace elision).
#include <cxxabi.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include "../drafter/drafter_pack.cuh"
#include "../glue/engine_kernels.cuh"
#include "../glue/glue.cuh"
#include "../glue/moe_dev.cuh"
#include "../glue/prefill_glue.cuh"
#include "../sample/sample.hpp"
#include "../decode_attn/decode_attn.cuh"
#include "../prefill_attn/prefill_attn.cuh"
#include "../prefill_moe/prefill_moe.cuh"

namespace {

struct Field {
  std::string name, type;
  size_t offset, size, align;
};
struct Struct {
  std::string name;
  size_t size, align;
  std::vector<Field> fields;
  size_t inits;    // initialisers the aggregate accepts (count_inits)
  size_t listed;   // initialisers the listed fields account for (1 per member, 1 per array element)
};

// converts to any member type (pointer, scalar, vector type, nested struct), never to an array: an array member takes
// one initialiser per element by brace elision
struct AnyInit {
  template <class U>
  operator U() const;
};

template <class T, class... A>
constexpr size_t count_inits() {
  if constexpr (requires { T{std::declval<A>()..., AnyInit{}}; })
    return count_inits<T, A..., AnyInit>();
  else
    return sizeof...(A);
}

template <class F>
constexpr size_t inits_of() {
  return sizeof(F) / sizeof(std::remove_all_extents_t<F>);   // 1 for a non-array, element count for an array
}

template <class T>
std::string type_name() {
  int st = 0;
  char* d = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &st);
  if (st != 0 || d == nullptr) {
    std::fprintf(stderr, "layout_emit: cannot demangle %s\n", typeid(T).name());
    std::exit(1);
  }
  std::string s(d);
  std::free(d);
  return s;
}

size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

// the listed fields, in offset order, must start exactly where alignment puts them and end where sizeof does
bool tiles(const Struct& s) {
  if (s.inits != s.listed) {
    std::fprintf(stderr,
                 "layout_emit: %s takes %zu initialisers, its listed fields account for %zu: a member is missing from "
                 "structs.def (or the struct is not an aggregate)\n",
                 s.name.c_str(), s.inits, s.listed);
    return false;
  }
  size_t end = 0;
  for (const Field& f : s.fields) {
    if (f.offset != align_up(end, f.align)) {
      std::fprintf(stderr, "layout_emit: %s.%s at %zu, expected %zu: a field before it is missing from structs.def\n",
                   s.name.c_str(), f.name.c_str(), f.offset, align_up(end, f.align));
      return false;
    }
    end = f.offset + f.size;
  }
  if (align_up(end, s.align) != s.size) {
    std::fprintf(stderr, "layout_emit: %s fields end at %zu, sizeof %zu: a trailing field is missing\n", s.name.c_str(),
                 end, s.size);
    return false;
  }
  return true;
}

std::vector<Struct> collect() {
  std::vector<Struct> out;
#define INK_F(f)                                                                                                      \
  s.fields.push_back(Field{#f, type_name<decltype(S::f)>(), offsetof(S, f), sizeof(S::f), alignof(decltype(S::f))});  \
  s.listed += inits_of<decltype(S::f)>();
#define INK_STRUCT(T, ...)                                                                                            \
  {                                                                                                                   \
    using S = T;                                                                                                      \
    static_assert(std::is_aggregate_v<S>, #T " must be an aggregate (the member count check needs it)");             \
    Struct s{#T, sizeof(S), alignof(S), {}, count_inits<S>(), 0};                                                     \
    __VA_ARGS__                                                                                                       \
    out.push_back(s);                                                                                                 \
  }
#include "structs.def"
#undef INK_STRUCT
#undef INK_F
  return out;
}

}  // namespace

int main() {
  const std::vector<Struct> all = collect();
  bool ok = true;
  for (const Struct& s : all) ok = tiles(s) && ok;
  if (!ok) return 1;
  std::printf("{\"format\": \"ink-kpack-layout/1\", \"structs\": [");
  for (size_t i = 0; i < all.size(); ++i) {
    const Struct& s = all[i];
    std::printf("%s\n {\"name\": \"%s\", \"size\": %zu, \"align\": %zu, \"fields\": [", i ? "," : "", s.name.c_str(),
                s.size, s.align);
    for (size_t j = 0; j < s.fields.size(); ++j) {
      const Field& f = s.fields[j];
      std::printf("%s\n  {\"name\": \"%s\", \"type\": \"%s\", \"offset\": %zu, \"size\": %zu, \"align\": %zu}",
                  j ? "," : "", f.name.c_str(), f.type.c_str(), f.offset, f.size, f.align);
    }
    std::printf("]}");
  }
  std::printf("\n]}\n");
  return 0;
}
