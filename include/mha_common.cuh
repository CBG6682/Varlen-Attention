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

// Type conversion helpers compatible with -D__CUDA_NO_HALF_CONVERSIONS__

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

// FP32 warp / sub-warp reductions

// Max across 4 contiguous lanes (for SM75 16x8x8 TiledMMA)
__device__ __forceinline__ float quad_reduce_max_f32(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

// Sum across 4 contiguous lanes (for Online Softmax on SM75 TiledMMA)
__device__ __forceinline__ float quad_reduce_sum_f32(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Max across 8 contiguous lanes (8 threads/row for head_dim = 64)
__device__ __forceinline__ float octet_reduce_max_f32(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

// Sum across 8 contiguous lanes (dot-product / rowsum dO*O for head_dim = 64)
__device__ __forceinline__ float octet_reduce_sum_f32(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Max across all 32 warp lanes
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Sum across all 32 warp lanes
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Non-interleaved RoPE helpers (first half [0..31] paired with second half [32..63])

// Forward RoPE R(theta) on FP32 element pair:
//   y0 = x0 * c0 - x1 * s0   (first half: d in [0..31])
//   y1 = x1 * c1 + x0 * s1   (second half: d + 32)
__device__ __forceinline__ void rope_fwd_pair_f32(
    float x0, float x1,
    float c0, float s0,
    float c1, float s1,
    float& y0, float& y1
) {
    y0 = fmaf(x0, c0, -x1 * s0);
    y1 = fmaf(x1, c1,  x0 * s1);
}

// Backward RoPE R(-theta) on FP32 gradient pair:
//   dx0 = g0 * c0 + g1 * s0  (first half: d in [0..31])
//   dx1 = g1 * c1 - g0 * s1  (second half: d + 32)
__device__ __forceinline__ void rope_bwd_pair_f32(
    float g0, float g1,
    float c0, float s0,
    float c1, float s1,
    float& dx0, float& dx1
) {
    dx0 = fmaf(g0, c0,  g1 * s0);
    dx1 = fmaf(g1, c1, -g0 * s1);
}

// Forward RoPE on uint4 g2r registers (8 contiguous FP16 elements per thread;
// 8 threads cover 1 row of 64 elements: col_id 0..3 = first half, 4..7 = second half).
// Uses __shfl_xor_sync(..., 4) to exchange the opposite half within the 8-lane group.
template <typename Element>
__device__ __forceinline__ void apply_rope_fwd_8elem_shfl(
    Element* __restrict__ reg_elems,    // 8 FP16 elements in thread registers
    const __half2* __restrict__ cos_h2, // 4 half2 cos values
    const __half2* __restrict__ sin_h2, // 4 half2 sin values
    bool is_second_half                 // true if thread handles columns 32..63
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

        // First half: x * cos - pair * sin | Second half: x * cos + pair * sin
        __half2 term1 = __hmul2(self_h2, c_h2);
        __half2 term2 = __hmul2(pair_h2, s_h2);
        __half2 res   = is_second_half ? __hadd2(term1, term2) : __hsub2(term1, term2);

        u32_ptr[i] = *reinterpret_cast<const uint32_t*>(&res);
    }
}
