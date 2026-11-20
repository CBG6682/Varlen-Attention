#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <limits>
#include <optional>
#include <tuple>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <torch/extension.h>

using namespace cute;

// =========================================================================================
// Các hàm chuyển đổi kiểu dữ liệu tương thích cờ -D__CUDA_NO_HALF_CONVERSIONS__
// =========================================================================================

template <typename InT>
__device__ __forceinline__ __half to_cuda_half(InT val) {
    return reinterpret_cast<const __half&>(val);
}

template <typename InT>
__device__ __forceinline__ float half_to_float(InT val) {
    return __half2float(reinterpret_cast<const __half&>(val));
}

template <typename OutT>
__device__ __forceinline__ OutT float_to_out(float val) {
    __half h = __float2half(val);
    return reinterpret_cast<const OutT&>(h);
}

template <typename OutT>
__device__ __forceinline__ OutT half_to_out(__half h) {
    return reinterpret_cast<const OutT&>(h);
}

// =========================================================================================
// Các hàm Warp / Sub-warp Reduction ở độ chính xác cao FP32 (Rule 7 & Rule 22)
// =========================================================================================

// Tìm Max trong nhóm 4 lane liên tiếp (dùng cho TiledMMA SM75 16x8x8 khi 1 warp giữ trọn hàng)
__device__ __forceinline__ float quad_reduce_max_f32(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

// Tính tổng trong nhóm 4 lane liên tiếp (dùng cho Online Softmax trên TiledMMA SM75)
__device__ __forceinline__ float quad_reduce_sum_f32(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Tìm Max trong nhóm 8 lane liên tiếp (dùng khi 8 threads/row xử lý head_dim = 64)
__device__ __forceinline__ float octet_reduce_max_f32(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

// Tính tổng trong nhóm 8 lane liên tiếp (dùng cho dot-product / rowsum dO*O với head_dim = 64)
__device__ __forceinline__ float octet_reduce_sum_f32(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Tìm Max trên toàn bộ 32 lane của Warp
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Tính tổng trên toàn bộ 32 lane của Warp
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// =========================================================================================
// Các hàm Helper cho Phép quay RoPE (Non-interleaved: nửa đầu [0..31] ghép nửa sau [32..63])
// =========================================================================================

// Xoay RoPE thuận R(theta) trực tiếp trên cặp phần tử FP32:
//   y0 = x0 * c0 - x1 * s0   (nửa đầu: d in [0..31])
//   y1 = x1 * c1 + x0 * s1   (nửa sau: d + 32)
__device__ __forceinline__ void rope_fwd_pair_f32(
    float x0, float x1,
    float c0, float s0,
    float c1, float s1,
    float& y0, float& y1
) {
    y0 = fmaf(x0, c0, -x1 * s0);
    y1 = fmaf(x1, c1,  x0 * s1);
}

// Xoay RoPE ngược R(-theta) trực tiếp trên cặp gradient FP32:
//   dx0 = g0 * c0 + g1 * s0  (nửa đầu: d in [0..31])
//   dx1 = g1 * c1 - g0 * s1  (nửa sau: d + 32)
__device__ __forceinline__ void rope_bwd_pair_f32(
    float g0, float g1,
    float c0, float s0,
    float c1, float s1,
    float& dx0, float& dx1
) {
    dx0 = fmaf(g0, c0,  g1 * s0);
    dx1 = fmaf(g1, c1, -g0 * s1);
}

// Xoay RoPE thuận trên thanh ghi g2r dạng uint4 (8 phần tử FP16 liên tiếp mỗi thread,
// 8 threads liên tiếp phụ trách trọn 1 hàng 64 phần tử: col_id = 0..3 là nửa đầu, 4..7 là nửa sau).
// Dùng __shfl_xor_sync(..., 4) để trao đổi nửa đối diện trong nội bộ nhóm 8 lane.
template <typename Element>
__device__ __forceinline__ void apply_rope_fwd_8elem_shfl(
    Element* __restrict__ reg_elems, // 8 phần tử FP16 trên thanh ghi của thread
    const __half2* __restrict__ cos_h2, // 4 half2 (8 phần tử cos của thread)
    const __half2* __restrict__ sin_h2, // 4 half2 (8 phần tử sin của thread)
    bool is_second_half                  // true nếu thread phụ trách cột 32..63
) {
    uint32_t* u32_ptr = reinterpret_cast<uint32_t*>(reg_elems);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t self_u32 = u32_ptr[i];
        uint32_t pair_u32 = __shfl_xor_sync(0xffffffff, self_u32, 4);

        __half2 self_h2 = *reinterpret_cast<const __half2*>(&self_u32);
        __half2 pair_h2 = *reinterpret_cast<const __half2*>(&pair_u32);

        __half2 c_h2 = cos_h2[i];
        __half2 s_h2 = sin_h2[i];

        // Nửa đầu: x * cos - pair * sin | Nửa sau: x * cos + pair * sin
        __half2 term1 = __hmul2(self_h2, c_h2);
        __half2 term2 = __hmul2(pair_h2, s_h2);
        __half2 res   = is_second_half ? __hadd2(term1, term2) : __hsub2(term1, term2);

        u32_ptr[i] = *reinterpret_cast<const uint32_t*>(&res);
    }
}
