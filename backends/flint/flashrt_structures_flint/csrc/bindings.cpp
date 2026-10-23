// SPDX-License-Identifier: Apache-2.0
//
// pybind surface for the Flint model kernels (module _flashrt_flint_ops). Pointers
// and streams are passed as integers, matching the FlashRT kernel library's
// convention, so the runtime can hand stable raw pointers to captured graphs.

#include "flint/flint_kernels.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace fl = flashrt_structures_flint;

namespace {

inline cudaStream_t to_stream(uintptr_t s) { return reinterpret_cast<cudaStream_t>(s); }
template <typename T>
inline T* ptr(uintptr_t p) { return reinterpret_cast<T*>(p); }

void check_launch(const char* op) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(e));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Flint (SM120) W4A16 NVFP4 kernels: repack, decode GEMV/MoE/attention, routing";

  m.attr("ARGMAX_BLOCKS") = fl::kArgmaxBlocks;
  // op-layer contract read by flashrt_structures_flint (the vLLM structures adapter): ABI, arch, weight layout,
  // and the routed/shared FFN profile (hidden, experts, top-k, expert width, shared width)
  py::dict tensor_api;
  tensor_api["abi"] = 1;
  tensor_api["capabilities"] = py::make_tuple(py::make_tuple(12, 0));
  tensor_api["layout"] = "e2m1_adjacent_e4m3_block16";
  tensor_api["routed_shared_profile"] = py::make_tuple(2048, 256, 6, 512, 1024);
  tensor_api["provider"] = "flashrt-structures-flint";
  tensor_api["package_version"] = "0.1.0";
  m.attr("TENSOR_API_METADATA") = tensor_api;

  m.def("gemv_w4a16",
        [](uintptr_t codes, uintptr_t scales, uintptr_t x, uintptr_t out, int N, int K, float gs,
           int epilogue, uintptr_t stream) {
          if (fl::gemv_w4a16(ptr<const void>(codes), ptr<const void>(scales), ptr<const void>(x),
                             ptr<void>(out), N, K, gs, epilogue, to_stream(stream)) != 0)
            throw std::invalid_argument("gemv_w4a16: unsupported K=" + std::to_string(K) +
                                        " epilogue=" + std::to_string(epilogue));
          check_launch("gemv_w4a16");
        });

  m.def("router_topk_bf16",
        [](uintptr_t xn, uintptr_t wr, uintptr_t logits, uintptr_t counter, uintptr_t ids, uintptr_t tw,
           uintptr_t stream) {
          fl::router_topk_bf16(ptr<const void>(xn), ptr<const void>(wr), ptr<float>(logits),
                               ptr<unsigned>(counter), ptr<int>(ids), ptr<float>(tw), to_stream(stream));
          check_launch("router_topk_bf16");
        });

  m.def("moe_gate_up_bf16",
        [](uintptr_t xn, uintptr_t ids, uintptr_t ecodes, uintptr_t escales, uintptr_t egs, uintptr_t scodes,
           uintptr_t sscales, float sgs, uintptr_t act, uintptr_t stream) {
          fl::moe_gate_up_bf16(ptr<const void>(xn), ptr<const int>(ids), ptr<const void>(ecodes),
                               ptr<const void>(escales), ptr<const float>(egs), ptr<const void>(scodes),
                               ptr<const void>(sscales), sgs, ptr<void>(act), to_stream(stream));
          check_launch("moe_gate_up_bf16");
        });

  m.def("moe_down_combine_bf16",
        [](uintptr_t act, uintptr_t ids, uintptr_t tw, uintptr_t ecodes, uintptr_t escales, uintptr_t egs,
           uintptr_t scodes, uintptr_t sscales, float sgs, uintptr_t out, uintptr_t stream) {
          fl::moe_down_combine_bf16(ptr<const void>(act), ptr<const int>(ids), ptr<const float>(tw),
                                    ptr<const void>(ecodes), ptr<const void>(escales), ptr<const float>(egs),
                                    ptr<const void>(scodes), ptr<const void>(sscales), sgs, ptr<void>(out),
                                    to_stream(stream));
          check_launch("moe_down_combine_bf16");
        });

  m.attr("EPI_BF16") = fl::kEpiBf16;
  m.attr("EPI_F32") = fl::kEpiF32;
  m.attr("EPI_SWIGLU_DENSE") = fl::kEpiSwigluDense;
  m.attr("EPI_SWIGLU_ROUTED") = fl::kEpiSwigluRouted;

  m.def("gemm_w4a16",
        [](uintptr_t A, int M, int lda, uintptr_t codes, uintptr_t scales, int N, int K, float gs, uintptr_t C, int ldc,
           int epilogue, uintptr_t stream) {
          if (fl::gemm_w4a16(ptr<const void>(A), M, lda, ptr<const void>(codes), ptr<const void>(scales), N, K, gs,
                             ptr<void>(C), ldc, epilogue, to_stream(stream)) != 0)
            throw std::invalid_argument("gemm_w4a16: unsupported N=" + std::to_string(N) + " K=" + std::to_string(K) +
                                        " epilogue=" + std::to_string(epilogue));
          check_launch("gemm_w4a16");
        });

  m.def("gemm_w4a16_grouped",
        [](uintptr_t A, uintptr_t gather, int lda, uintptr_t tiles, uintptr_t n_tiles, int max_tiles, uintptr_t codes,
           uintptr_t scales, int N, int K, uintptr_t gs_e, uintptr_t C, int ldc, int epilogue, int tile_rows,
           int mean_rows, uintptr_t stream) {
          if (fl::gemm_w4a16_grouped(ptr<const void>(A), ptr<const int>(gather), lda, ptr<const int>(tiles),
                                     ptr<const int>(n_tiles), max_tiles, ptr<const void>(codes), ptr<const void>(scales),
                                     N, K, ptr<const float>(gs_e), ptr<void>(C), ldc, epilogue, tile_rows,
                                     mean_rows, to_stream(stream)) != 0)
            throw std::invalid_argument("gemm_w4a16_grouped: unsupported N=" + std::to_string(N) + " K=" +
                                        std::to_string(K) + " epilogue=" + std::to_string(epilogue) +
                                        " tile_rows=" + std::to_string(tile_rows));
          check_launch("gemm_w4a16_grouped");
        });

  m.def("gemm_w4a16_grouped_q8",
        [](uintptr_t A, uintptr_t gather, int lda, uintptr_t tiles, uintptr_t n_tiles, int max_tiles, uintptr_t codes,
           uintptr_t scales, int N, int K, uintptr_t gs_e, uintptr_t C, int ldc, uintptr_t sq, int tile_rows,
           int mean_rows, uintptr_t stream) {
          if (fl::gemm_w4a16_grouped_q8(ptr<const void>(A), ptr<const int>(gather), lda, ptr<const int>(tiles),
                                        ptr<const int>(n_tiles), max_tiles, ptr<const void>(codes),
                                        ptr<const void>(scales), N, K, ptr<const float>(gs_e), ptr<void>(C), ldc,
                                        ptr<float>(sq), tile_rows, mean_rows, to_stream(stream)) != 0)
            throw std::invalid_argument("gemm_w4a16_grouped_q8: unsupported configuration");
        });

  m.def("gemm_w4a16_splitk",
        [](uintptr_t A, int M, int lda, uintptr_t codes, uintptr_t scales, int N, int K, float gs, uintptr_t part,
           uintptr_t C, int ldc, int epilogue, int bm, int stages, int split, int kb, uintptr_t stream) {
          if (fl::gemm_w4a16_splitk(ptr<const void>(A), M, lda, ptr<const void>(codes), ptr<const void>(scales), N, K,
                                    gs, ptr<float>(part), ptr<void>(C), ldc, epilogue,
                                    fl::GemmConfig{bm, stages, split, kb}, to_stream(stream)) != 0)
            throw std::invalid_argument("gemm_w4a16_splitk: unsupported configuration");
        });

  // ---- W4A8 ----
  m.def("quant_rows_s8", [](uintptr_t x, int M, int K, int ldx, uintptr_t q, uintptr_t s, uintptr_t stream) {
    fl::quant_rows_s8(ptr<const void>(x), M, K, ldx, ptr<void>(q), ptr<float>(s), to_stream(stream));
  });
  m.def("quant_rows_s8_perrow", [](uintptr_t x, int M, int K, int ldx, uintptr_t q, uintptr_t s, uintptr_t stream) {
    fl::quant_rows_s8_perrow(ptr<const void>(x), M, K, ldx, ptr<void>(q), ptr<float>(s), to_stream(stream));
  });
  m.def("gemv_w4a16_rows",
        [](uintptr_t codes, uintptr_t scales, uintptr_t x, int M, uintptr_t out, int N, int K, float gs, int epilogue,
           uintptr_t stream) {
          if (fl::gemv_w4a16_rows(ptr<const void>(codes), ptr<const void>(scales), ptr<const void>(x), M, ptr<void>(out),
                                  N, K, gs, epilogue, to_stream(stream)) != 0)
            throw std::invalid_argument("gemv_w4a16_rows: unsupported M/K/epilogue");
        });
  m.def("gemm_s8_resident",
        [](uintptr_t A, uintptr_t AS, int M, int K, uintptr_t codes, uintptr_t scales, int N, int KS, float gs,
           uintptr_t part, uintptr_t C, int ldc, int epilogue, uintptr_t stream, uintptr_t dbg) {
          if (fl::gemm_s8_resident(ptr<const void>(A), ptr<const void>(AS), M, K, ptr<const void>(codes),
                                   ptr<const void>(scales), N, KS, gs, ptr<void>(part), ptr<void>(C), ldc, epilogue,
                                   to_stream(stream), ptr<unsigned long long>(dbg)) != 0)
            throw std::invalid_argument("gemm_s8_resident: unsupported configuration");
        });
  m.def("gemm_resident_w4a16",
        [](uintptr_t x, int M, int K, uintptr_t codes, uintptr_t scales, int N, int KS, float gs, uintptr_t part,
           uintptr_t C, int ldc, int epilogue, uintptr_t stream) {
          if (fl::gemm_resident_w4a16(ptr<const void>(x), M, K, ptr<const void>(codes), ptr<const void>(scales), N, KS,
                                      gs, ptr<float>(part), ptr<void>(C), ldc, epilogue, to_stream(stream)) != 0)
            throw std::invalid_argument("gemm_resident_w4a16: unsupported configuration");
        });
  m.def("gemm_stream_w4a16",
        [](uintptr_t codes, uintptr_t scales, uintptr_t x, int M, uintptr_t out, int N, int K, float gs, int epilogue,
           uintptr_t stream, uintptr_t dbg) {
          if (fl::gemm_stream_w4a16(ptr<const void>(codes), ptr<const void>(scales), ptr<const void>(x), M, ptr<void>(out),
                                    N, K, gs, epilogue, to_stream(stream), ptr<unsigned long long>(dbg)) != 0)
            throw std::invalid_argument("gemm_stream_w4a16: unsupported M/N/K/epilogue");
        });
  m.def("add_rmsnorm_rows_bf16",
        [](uintptr_t x, uintptr_t res_in, uintptr_t w, float eps, uintptr_t res_out, uintptr_t out, int M,
           uintptr_t stream) {
          fl::add_rmsnorm_rows_bf16(ptr<const void>(x), ptr<const void>(res_in), ptr<const float>(w), eps,
                                    ptr<void>(res_out), ptr<void>(out), M, to_stream(stream));
        });
  m.def("gemv_mma_norm_w4a16",
        [](uintptr_t codes, uintptr_t scales, uintptr_t x, uintptr_t res_in, uintptr_t res_out, uintptr_t norm_w,
           float eps, int M, uintptr_t out, int N, int K, float gs, int epilogue, uintptr_t stream) {
          if (fl::gemv_mma_norm_w4a16(ptr<const void>(codes), ptr<const void>(scales), ptr<const void>(x),
                                      ptr<const void>(res_in), ptr<void>(res_out), ptr<const float>(norm_w), eps, M,
                                      ptr<void>(out), N, K, gs, epilogue, to_stream(stream)) != 0)
            throw std::invalid_argument("gemv_mma_norm_w4a16: unsupported M/K/epilogue");
        });
  m.def("gemv_mma_w4a16",
        [](uintptr_t codes, uintptr_t scales, uintptr_t x, int M, uintptr_t out, int N, int K, float gs, int epilogue,
           uintptr_t stream) {
          if (fl::gemv_mma_w4a16(ptr<const void>(codes), ptr<const void>(scales), ptr<const void>(x), M, ptr<void>(out),
                                 N, K, gs, epilogue, to_stream(stream)) != 0)
            throw std::invalid_argument("gemv_mma_w4a16: unsupported M/K/epilogue");
        });
  m.def("decode_attention_paged_bf16",
        [](uintptr_t q, uintptr_t k_cache, uintptr_t v_cache, long long sb, long long st, long long sh, int block_size,
           uintptr_t block_tables, int bt_stride, uintptr_t seq_lens, int n_seq, int window, uintptr_t sinks,
           float scale, int n_split, uintptr_t o_part, uintptr_t ml_part, uintptr_t out, uintptr_t stream,
           int kv_fp8, uintptr_t v4c, uintptr_t v4s, long long v4cb, long long v4ct, long long v4ch, long long v4sb, long long v4st, long long v4sh) {
          fl::V4Shadow v4{ptr<const void>(v4c), ptr<const void>(v4s), v4cb, v4ct, v4ch, v4sb, v4st, v4sh};
          fl::decode_attention_paged_bf16(ptr<const void>(q), ptr<const void>(k_cache), ptr<const void>(v_cache), sb,
                                          st, sh, block_size, ptr<const int>(block_tables), bt_stride,
                                          ptr<const int>(seq_lens), n_seq, window, ptr<const float>(sinks), scale,
                                          n_split, ptr<float>(o_part), ptr<float>(ml_part), ptr<void>(out),
                                          to_stream(stream), kv_fp8, v4c ? &v4 : nullptr);
        });
  m.def("router_topk_small_bf16",
        [](uintptr_t xn, int n, uintptr_t wr, uintptr_t logits, uintptr_t ids, uintptr_t tw, uintptr_t counts,
           uintptr_t stream) {
          if (fl::router_topk_small_bf16(ptr<const void>(xn), n, ptr<const void>(wr), ptr<float>(logits), ptr<int>(ids),
                                         ptr<float>(tw), ptr<int>(counts), to_stream(stream)) != 0)
            throw std::invalid_argument("router_topk_small_bf16: n must be in [1, 32]");
        });
  m.def("decode_attention_paged_qkv_bf16",
        [](uintptr_t qkv, long long qkv_stride, uintptr_t positions, uintptr_t q_w, uintptr_t k_w, uintptr_t cos_sin,
           long long cs_stride, float eps, uintptr_t k_cache, uintptr_t v_cache, long long sb, long long st,
           long long sh, int block_size, uintptr_t block_tables, int bt_stride, uintptr_t seq_lens, int n_seq,
           int window, uintptr_t sinks, float scale, int n_split, uintptr_t o_part, uintptr_t ml_part, uintptr_t out,
           uintptr_t stream, int kv_fp8, uintptr_t v4c, uintptr_t v4s, long long v4cb, long long v4ct, long long v4ch, long long v4sb, long long v4st, long long v4sh) {
          fl::V4Shadow v4{ptr<const void>(v4c), ptr<const void>(v4s), v4cb, v4ct, v4ch, v4sb, v4st, v4sh};
          fl::decode_attention_paged_qkv_bf16(ptr<const void>(qkv), qkv_stride, ptr<const long long>(positions),
                                              ptr<const float>(q_w), ptr<const float>(k_w), ptr<const float>(cos_sin),
                                              cs_stride, eps, ptr<const void>(k_cache), ptr<const void>(v_cache), sb, st,
                                              sh, block_size, ptr<const int>(block_tables), bt_stride,
                                              ptr<const int>(seq_lens), n_seq, window, ptr<const float>(sinks), scale,
                                              n_split, ptr<float>(o_part), ptr<float>(ml_part), ptr<void>(out),
                                              to_stream(stream), kv_fp8, v4c ? &v4 : nullptr);
        });
  m.def("gemm_w4a8",
        [](uintptr_t A, uintptr_t AS, int M, uintptr_t codes, uintptr_t scales, int N, int K, float gs, uintptr_t C,
           int ldc, int epilogue, uintptr_t stream) {
          if (fl::gemm_w4a8(ptr<const void>(A), ptr<const void>(AS), M, ptr<const void>(codes), ptr<const void>(scales),
                            N, K, gs, ptr<void>(C), ldc, epilogue, to_stream(stream)) != 0)
            throw std::invalid_argument("gemm_w4a8: unsupported N=" + std::to_string(N) + " K=" + std::to_string(K) +
                                        " epilogue=" + std::to_string(epilogue));
        });
  m.def("gemm_w4a8_cfg",
        [](uintptr_t A, uintptr_t AS, int M, uintptr_t codes, uintptr_t scales, int N, int K, float gs, uintptr_t C,
           int ldc, int epilogue, int bm, int stages, int split, int kb, uintptr_t stream) {
          if (fl::gemm_w4a8_cfg(ptr<const void>(A), ptr<const void>(AS), M, ptr<const void>(codes),
                                ptr<const void>(scales), N, K, gs, ptr<void>(C), ldc, epilogue,
                                fl::GemmConfig{bm, stages, split, kb}, to_stream(stream)) != 0)
            throw std::invalid_argument("gemm_w4a8_cfg: unsupported configuration");
        });
  m.def("gemm_w4a8_grouped",
        [](uintptr_t A, uintptr_t AS, uintptr_t gather, uintptr_t tiles, uintptr_t n_tiles, int max_tiles,
           uintptr_t codes, uintptr_t scales, int N, int K, uintptr_t gs_e, uintptr_t C, int ldc, int epilogue,
           int tile_rows, int mean_rows, uintptr_t stream) {
          if (fl::gemm_w4a8_grouped(ptr<const void>(A), ptr<const void>(AS), ptr<const int>(gather),
                                    ptr<const int>(tiles), ptr<const int>(n_tiles), max_tiles, ptr<const void>(codes),
                                    ptr<const void>(scales), N, K, ptr<const float>(gs_e), ptr<void>(C), ldc,
                                    epilogue, tile_rows, mean_rows, to_stream(stream)) != 0)
            throw std::invalid_argument("gemm_w4a8_grouped: unsupported N=" + std::to_string(N) + " K=" +
                                        std::to_string(K) + " epilogue=" + std::to_string(epilogue) +
                                        " tile_rows=" + std::to_string(tile_rows));
        });
  m.def("gemm_w4a8_narrow",
        [](uintptr_t A, uintptr_t AS, int M, uintptr_t codes, uintptr_t scales, int N, int K, float gs, uintptr_t C,
           int ldc, int epilogue, uintptr_t stream) {
          if (fl::gemm_w4a8_narrow(ptr<const void>(A), ptr<const void>(AS), M, ptr<const void>(codes),
                                   ptr<const void>(scales), N, K, gs, ptr<void>(C), ldc, epilogue,
                                   to_stream(stream)) != 0)
            throw std::invalid_argument("gemm_w4a8_narrow: not applicable");
        });
  m.def("set_gemm_s8_override", [](int bm, int stages, int split, int kb, int gbm, int gstages, int gsplit, int gkb) {
    fl::set_gemm_s8_override(fl::GemmConfig{bm, stages, split, kb}, fl::GemmConfig{gbm, gstages, gsplit, gkb});
  });

  m.def("router_topk_rows_bf16",
        [](uintptr_t xn, int n, uintptr_t wr, uintptr_t ids, uintptr_t tw, uintptr_t counts, uintptr_t pos,
           uintptr_t n_end, uintptr_t stream) {
          fl::router_topk_rows_bf16(ptr<const void>(xn), n, ptr<const void>(wr), ptr<int>(ids), ptr<float>(tw),
                                    ptr<int>(counts), ptr<const int>(pos), ptr<const int>(n_end), to_stream(stream));
          check_launch("router_topk_rows_bf16");
        });

  m.def("moe_scan_place",
        [](uintptr_t ids, int n, int tile_rows, uintptr_t counts, uintptr_t sorted_tok, uintptr_t inv, uintptr_t tiles,
           uintptr_t n_tiles, uintptr_t pos, uintptr_t n_end, uintptr_t stream) {
          fl::moe_scan_place(ptr<const int>(ids), n, tile_rows, ptr<int>(counts), ptr<int>(sorted_tok), ptr<int>(inv),
                             ptr<int>(tiles), ptr<int>(n_tiles), ptr<const int>(pos), ptr<const int>(n_end),
                             to_stream(stream));
          check_launch("moe_scan_place");
        });

  m.def("moe_combine_rows_bf16",
        [](uintptr_t y, uintptr_t inv, uintptr_t tw, uintptr_t shared, uintptr_t out, int n, uintptr_t stream) {
          fl::moe_combine_rows_bf16(ptr<const void>(y), ptr<const int>(inv), ptr<const float>(tw),
                                    ptr<const void>(shared), ptr<void>(out), n, to_stream(stream));
          check_launch("moe_combine_rows_bf16");
        });

  // ---- fused decode step (PDL) ----
  m.def("gemv_fused_w4a16",
        [](uintptr_t codes, uintptr_t scales, uintptr_t x, uintptr_t res_in, uintptr_t res_out, uintptr_t norm_w,
           int first, float eps, uintptr_t out, int N, int K, float gs, int epilogue, uintptr_t stream) {
          if (fl::gemv_fused_w4a16(ptr<const void>(codes), ptr<const void>(scales), ptr<const void>(x),
                              ptr<const void>(res_in), ptr<void>(res_out), ptr<const float>(norm_w), first, eps,
                              ptr<void>(out), N, K, gs, epilogue, to_stream(stream)) != 0)
            throw std::invalid_argument("gemv_fused_w4a16: unsupported K=" + std::to_string(K) + " epilogue=" +
                                        std::to_string(epilogue) + (norm_w ? " (fused norm)" : ""));
          check_launch("gemv_fused_w4a16");
        });

  m.def("router_norm_bf16",
        [](uintptr_t x, uintptr_t res_in, uintptr_t nw, float eps, uintptr_t res_out, uintptr_t xn_out, uintptr_t wr,
           uintptr_t logits, uintptr_t counter, uintptr_t ids, uintptr_t tw, uintptr_t stream) {
          fl::router_norm_bf16(ptr<const void>(x), ptr<const void>(res_in), ptr<const float>(nw), eps, ptr<void>(res_out),
                               ptr<void>(xn_out), ptr<const void>(wr), ptr<float>(logits), ptr<unsigned>(counter),
                               ptr<int>(ids), ptr<float>(tw), to_stream(stream));
          check_launch("router_norm_bf16");
        });
  m.def("moe_gate_up_pdl_bf16",
        [](uintptr_t xn, uintptr_t ids, uintptr_t ecodes, uintptr_t escales, uintptr_t egs, uintptr_t scodes,
           uintptr_t sscales, float sgs, uintptr_t act, uintptr_t stream) {
          fl::moe_gate_up_pdl_bf16(ptr<const void>(xn), ptr<const int>(ids), ptr<const void>(ecodes), ptr<const void>(escales),
                                ptr<const float>(egs), ptr<const void>(scodes), ptr<const void>(sscales), sgs,
                                ptr<void>(act), to_stream(stream));
          check_launch("moe_gate_up_pdl_bf16");
        });
  m.def("moe_down_combine_pdl_bf16",
        [](uintptr_t act, uintptr_t ids, uintptr_t tw, uintptr_t ecodes, uintptr_t escales, uintptr_t egs,
           uintptr_t scodes, uintptr_t sscales, float sgs, uintptr_t out, uintptr_t stream) {
          fl::moe_down_combine_pdl_bf16(ptr<const void>(act), ptr<const int>(ids), ptr<const float>(tw),
                                     ptr<const void>(ecodes), ptr<const void>(escales), ptr<const float>(egs),
                                     ptr<const void>(scodes), ptr<const void>(sscales), sgs, ptr<void>(out),
                                     to_stream(stream));
          check_launch("moe_down_combine_pdl_bf16");
        });

}
