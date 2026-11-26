// Paged KV Cache management CUDA/CuTe kernels (PAGE_SIZE = 16, HEAD_DIM = 64)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "mha_common.cuh"
#include "mha_ops.cuh"

using namespace cute;

// =================================================================================================
// 1. Device Kernels
// =================================================================================================

template <
    int PAGE_SIZE = 16,
    int HEAD_DIM = 64,
    int NUM_THREADS = 128
>
__global__ void __launch_bounds__(NUM_THREADS)
kernel_prefill_to_paged_kv(
    const cutlass::half_t* __restrict__ K_ptr,    // (Total_Lk, H_kv, 64) [FP16]
    const cutlass::half_t* __restrict__ V_ptr,    // (Total_Lk, H_kv, 64) [FP16]
    const int32_t* __restrict__ cu_seqlens_k_ptr, // (B + 1,) [Int32]
    const int32_t* __restrict__ block_table_ptr,  // (B, max_pages_per_seq) [Int32]
    cutlass::half_t* __restrict__ K_cache_ptr,    // (num_pages, 16, H_kv, 64) [FP16]
    cutlass::half_t* __restrict__ V_cache_ptr,    // (num_pages, 16, H_kv, 64) [FP16]
    int32_t* __restrict__ cache_seqlens_ptr,      // (B,) [Int32]
    int64_t k_stride_l,                           // scalar: int64
    int64_t k_stride_h,                           // scalar: int64
    int64_t v_stride_l,                           // scalar: int64
    int64_t v_stride_h,                           // scalar: int64
    int H_kv,                                     // scalar: int
    int max_pages_per_seq                         // scalar: int
) {
    static_assert(PAGE_SIZE * (HEAD_DIM / 8) == NUM_THREADS, "Thread layout must cover one 16x64 page");

    // 1. Coordinates & sequence bounds
    const int tid      = static_cast<int>(threadIdx.x);
    const int page_idx = static_cast<int>(blockIdx.x);
    const int h_kv     = static_cast<int>(blockIdx.y);
    const int seq_idx  = static_cast<int>(blockIdx.z);

    const int k_start = cu_seqlens_k_ptr[seq_idx];
    const int k_end   = cu_seqlens_k_ptr[seq_idx + 1];
    const int lk_i    = k_end - k_start;

    // Record cache_seqlens[seq_idx] before early exit so empty sequences are initialized
    if (page_idx == 0 && h_kv == 0) {
        using TiledCopyStoreLen = decltype(make_tiled_copy(
            Copy_Atom<UniversalCopy<int32_t>, int32_t>{},
            make_layout(Shape<Int<NUM_THREADS>, _1>{}, LayoutRight{}),
            make_layout(Shape<_1, _1>{}, LayoutRight{})));
        TiledCopyStoreLen tiled_copy_len;
        auto thr_copy_len = tiled_copy_len.get_thread_slice(tid);
        Tensor gLen_bcast = make_tensor(
            make_gmem_ptr(cache_seqlens_ptr + seq_idx),
            make_shape(Int<NUM_THREADS>{}, _1{}),
            make_stride(_0{}, _1{}));
        Tensor tLgLen = thr_copy_len.partition_D(gLen_bcast);
        Tensor rLen = make_fragment_like(tLgLen);
        rLen(0) = lk_i;
        Tensor tPredLen = make_tensor<bool>(make_shape(size<1>(tLgLen), size<2>(tLgLen)));
        tPredLen(0, 0) = (tid == 0);
        cute::copy_if(tiled_copy_len, tPredLen, rLen, tLgLen);
    }

    const int token_start = page_idx * PAGE_SIZE;
    if (token_start >= lk_i) {
        return;
    }

    const int physical_page = block_table_ptr[static_cast<int64_t>(seq_idx) * max_pages_per_seq + page_idx];
    const int valid_rows = min(PAGE_SIZE, lk_i - token_start);
    const bool page_in_bounds = (valid_rows == PAGE_SIZE);

    // 2. CuTe 128-bit GMEM <-> Reg TiledCopy partitioning
    using Element = cutlass::half_t;
    using TiledCopyKV = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint4>, Element>{},
        make_layout(Shape<Int<PAGE_SIZE>, Int<HEAD_DIM / 8>>{}, LayoutRight{}),
        make_layout(Shape<_1, _8>{}, LayoutRight{})));

    TiledCopyKV tiled_copy_kv;
    auto thr_copy_kv = tiled_copy_kv.get_thread_slice(tid);

    const int64_t src_tok = static_cast<int64_t>(k_start + token_start);
    const int64_t k_src_off = src_tok * k_stride_l + static_cast<int64_t>(h_kv) * k_stride_h;
    const int64_t v_src_off = src_tok * v_stride_l + static_cast<int64_t>(h_kv) * v_stride_h;
    const int64_t dst_off =
        (static_cast<int64_t>(physical_page) * PAGE_SIZE * H_kv + h_kv) * HEAD_DIM;
    const int64_t dst_row_stride = static_cast<int64_t>(H_kv) * HEAD_DIM;

    Tensor gK_src = make_tensor(
        make_gmem_ptr(K_ptr + k_src_off),
        make_shape(Int<PAGE_SIZE>{}, Int<HEAD_DIM>{}),
        make_stride(k_stride_l, _1{}));
    Tensor gV_src = make_tensor(
        make_gmem_ptr(V_ptr + v_src_off),
        make_shape(Int<PAGE_SIZE>{}, Int<HEAD_DIM>{}),
        make_stride(v_stride_l, _1{}));
    Tensor gK_dst = make_tensor(
        make_gmem_ptr(K_cache_ptr + dst_off),
        make_shape(Int<PAGE_SIZE>{}, Int<HEAD_DIM>{}),
        make_stride(dst_row_stride, _1{}));
    Tensor gV_dst = make_tensor(
        make_gmem_ptr(V_cache_ptr + dst_off),
        make_shape(Int<PAGE_SIZE>{}, Int<HEAD_DIM>{}),
        make_stride(dst_row_stride, _1{}));

    Tensor tKs = thr_copy_kv.partition_S(gK_src);
    Tensor tVs = thr_copy_kv.partition_S(gV_src);
    Tensor tKd = thr_copy_kv.partition_D(gK_dst);
    Tensor tVd = thr_copy_kv.partition_D(gV_dst);

    Tensor rK = make_fragment_like(tKd);
    Tensor rV = make_fragment_like(tVd);
    CUTE_STATIC_ASSERT_V(size(rK) == Int<8>{});

    Tensor cTile = thr_copy_kv.partition_D(
        make_identity_tensor(make_shape(Int<PAGE_SIZE>{}, Int<HEAD_DIM>{})));
    Tensor tPred = make_tensor<bool>(make_shape(size<1>(tKd), size<2>(tKd)));
    tPred(0, 0) = (get<0>(cTile(0, 0, 0)) < valid_rows);

    // 3. Vectorized 128-bit load -> store into paged KV pool
    if (page_in_bounds) {
        cute::copy(tiled_copy_kv, tKs, rK);
        cute::copy(tiled_copy_kv, tVs, rV);
        cute::copy(tiled_copy_kv, rK, tKd);
        cute::copy(tiled_copy_kv, rV, tVd);
    } else {
        clear(rK);
        clear(rV);
        cute::copy_if(tiled_copy_kv, tPred, tKs, rK);
        cute::copy_if(tiled_copy_kv, tPred, tVs, rV);
        cute::copy_if(tiled_copy_kv, tPred, rK, tKd);
        cute::copy_if(tiled_copy_kv, tPred, rV, tVd);
    }
}

template <
    int PAGE_SIZE = 16,
    int HEAD_DIM = 64,
    int HEAD_TILE = 16,
    int NUM_THREADS = 128
>
__global__ void __launch_bounds__(NUM_THREADS)
kernel_paged_kv_append(
    const cutlass::half_t* __restrict__ new_K_ptr, // (B, H_kv, 64) [FP16]
    const cutlass::half_t* __restrict__ new_V_ptr, // (B, H_kv, 64) [FP16]
    const int32_t* __restrict__ block_table_ptr,   // (B, max_pages_per_seq) [Int32]
    cutlass::half_t* __restrict__ K_cache_ptr,     // (num_pages, 16, H_kv, 64) [FP16]
    cutlass::half_t* __restrict__ V_cache_ptr,     // (num_pages, 16, H_kv, 64) [FP16]
    int32_t* __restrict__ cache_seqlens_ptr,       // (B,) [Int32]
    int64_t k_stride_b,                            // scalar: int64
    int64_t k_stride_h,                            // scalar: int64
    int64_t v_stride_b,                            // scalar: int64
    int64_t v_stride_h,                            // scalar: int64
    int H_kv,                                      // scalar: int
    int max_pages_per_seq                          // scalar: int
) {
    static_assert(HEAD_TILE * (HEAD_DIM / 8) == NUM_THREADS, "Thread layout must cover 16x64 head tile");

    // 1. Sequence slot lookup & CuTe setup
    const int tid     = static_cast<int>(threadIdx.x);
    const int seq_idx = static_cast<int>(blockIdx.x);

    const int pos           = cache_seqlens_ptr[seq_idx];
    const int logical_page  = pos / PAGE_SIZE;
    const int slot          = pos % PAGE_SIZE;
    const int physical_page = block_table_ptr[static_cast<int64_t>(seq_idx) * max_pages_per_seq + logical_page];

    using Element = cutlass::half_t;
    using TiledCopyHead = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint4>, Element>{},
        make_layout(Shape<Int<HEAD_TILE>, Int<HEAD_DIM / 8>>{}, LayoutRight{}),
        make_layout(Shape<_1, _8>{}, LayoutRight{})));
    using TiledCopyStoreLen = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<int32_t>, int32_t>{},
        make_layout(Shape<Int<NUM_THREADS>, _1>{}, LayoutRight{}),
        make_layout(Shape<_1, _1>{}, LayoutRight{})));

    TiledCopyHead tiled_copy_head;
    auto thr_copy_head = tiled_copy_head.get_thread_slice(tid);

    TiledCopyStoreLen tiled_copy_len;
    auto thr_copy_len = tiled_copy_len.get_thread_slice(tid);

    Tensor gLen_bcast = make_tensor(
        make_gmem_ptr(cache_seqlens_ptr + seq_idx),
        make_shape(Int<NUM_THREADS>{}, _1{}),
        make_stride(_0{}, _1{}));
    Tensor tLgLen = thr_copy_len.partition_D(gLen_bcast);
    Tensor rLen = make_fragment_like(tLgLen);
    rLen(0) = pos + 1;
    Tensor tPredLen = make_tensor<bool>(make_shape(size<1>(tLgLen), size<2>(tLgLen)));
    tPredLen(0, 0) = (tid == 0);

    Tensor gCacheDummy = make_tensor(
        make_gmem_ptr(K_cache_ptr),
        make_shape(Int<HEAD_TILE>{}, Int<HEAD_DIM>{}),
        make_stride(Int<HEAD_DIM>{}, _1{}));
    Tensor tDummyD = thr_copy_head.partition_D(gCacheDummy);
    Tensor rK = make_fragment_like(tDummyD);
    Tensor rV = make_fragment_like(tDummyD);
    CUTE_STATIC_ASSERT_V(size(rK) == Int<8>{});

    Tensor cTile = thr_copy_head.partition_D(
        make_identity_tensor(make_shape(Int<HEAD_TILE>{}, Int<HEAD_DIM>{})));
    const int row_in_tile = get<0>(cTile(0, 0, 0));
    Tensor tPred = make_tensor<bool>(make_shape(size<1>(tDummyD), size<2>(tDummyD)));

    const int64_t k_src_base = static_cast<int64_t>(seq_idx) * k_stride_b;
    const int64_t v_src_base = static_cast<int64_t>(seq_idx) * v_stride_b;
    const int64_t cache_slot_base =
        (static_cast<int64_t>(physical_page) * PAGE_SIZE + slot) * H_kv * HEAD_DIM;

    // 2. Copy all H_kv heads of token seq_idx into paged slot in chunks of HEAD_TILE
    for (int h_base = 0; h_base < H_kv; h_base += HEAD_TILE) {
        const int valid_heads = H_kv - h_base;
        const bool full_tile = (valid_heads >= HEAD_TILE);
        if (!full_tile) {
            tPred(0, 0) = (row_in_tile < valid_heads);
        }

        Tensor gK_src = make_tensor(
            make_gmem_ptr(new_K_ptr + k_src_base + static_cast<int64_t>(h_base) * k_stride_h),
            make_shape(Int<HEAD_TILE>{}, Int<HEAD_DIM>{}),
            make_stride(k_stride_h, _1{}));
        Tensor gV_src = make_tensor(
            make_gmem_ptr(new_V_ptr + v_src_base + static_cast<int64_t>(h_base) * v_stride_h),
            make_shape(Int<HEAD_TILE>{}, Int<HEAD_DIM>{}),
            make_stride(v_stride_h, _1{}));
        Tensor gK_dst = make_tensor(
            make_gmem_ptr(K_cache_ptr + cache_slot_base + static_cast<int64_t>(h_base) * HEAD_DIM),
            make_shape(Int<HEAD_TILE>{}, Int<HEAD_DIM>{}),
            make_stride(Int<HEAD_DIM>{}, _1{}));
        Tensor gV_dst = make_tensor(
            make_gmem_ptr(V_cache_ptr + cache_slot_base + static_cast<int64_t>(h_base) * HEAD_DIM),
            make_shape(Int<HEAD_TILE>{}, Int<HEAD_DIM>{}),
            make_stride(Int<HEAD_DIM>{}, _1{}));

        Tensor tKs = thr_copy_head.partition_S(gK_src);
        Tensor tVs = thr_copy_head.partition_S(gV_src);
        Tensor tKd = thr_copy_head.partition_D(gK_dst);
        Tensor tVd = thr_copy_head.partition_D(gV_dst);

        if (full_tile) {
            cute::copy(tiled_copy_head, tKs, rK);
            cute::copy(tiled_copy_head, tVs, rV);
            cute::copy(tiled_copy_head, rK, tKd);
            cute::copy(tiled_copy_head, rV, tVd);
        } else {
            clear(rK);
            clear(rV);
            cute::copy_if(tiled_copy_head, tPred, tKs, rK);
            cute::copy_if(tiled_copy_head, tPred, tVs, rV);
            cute::copy_if(tiled_copy_head, tPred, rK, tKd);
            cute::copy_if(tiled_copy_head, tPred, rV, tVd);
        }
    }

    // 3. Synchronize CTA and increment cache_seqlens[seq_idx] atomically within block
    __syncthreads();
    cute::copy_if(tiled_copy_len, tPredLen, rLen, tLgLen);
}

// =================================================================================================
// 2. Host Functions
// =================================================================================================

void launch_prefill_to_paged_kv(
    const at::Tensor& k,             // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& v,             // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& cu_seqlens_k,  // (B + 1,) [Int32]
    at::Tensor& k_cache,             // (num_pages, 16, H_kv, 64) [FP16]
    at::Tensor& v_cache,             // (num_pages, 16, H_kv, 64) [FP16]
    const at::Tensor& block_table,   // (B, max_pages_per_seq) [Int32]
    at::Tensor& cache_seqlens,       // (B,) [Int32]
    int64_t max_seqlen_k             // scalar: int64
) {
    // 1. Input validation
    TORCH_CHECK(k.is_cuda() && v.is_cuda() && cu_seqlens_k.is_cuda(),
                "k, v, and cu_seqlens_k must be CUDA tensors");
    TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda() && block_table.is_cuda() && cache_seqlens.is_cuda(),
                "k_cache, v_cache, block_table, and cache_seqlens must be CUDA tensors");
    TORCH_CHECK(
        v.device() == k.device() && cu_seqlens_k.device() == k.device() &&
        k_cache.device() == k.device() && v_cache.device() == k.device() &&
        block_table.device() == k.device() && cache_seqlens.device() == k.device(),
        "All tensors must reside on the same CUDA device"
    );

    TORCH_CHECK(
        cu_seqlens_k.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous() &&
        block_table.is_contiguous() && cache_seqlens.is_contiguous(),
        "cu_seqlens_k, k_cache, v_cache, block_table, and cache_seqlens must be contiguous"
    );
    TORCH_CHECK(
        k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf &&
        k_cache.scalar_type() == at::kHalf && v_cache.scalar_type() == at::kHalf,
        "k, v, k_cache, and v_cache must be float16"
    );
    TORCH_CHECK(
        cu_seqlens_k.scalar_type() == at::kInt &&
        block_table.scalar_type() == at::kInt &&
        cache_seqlens.scalar_type() == at::kInt,
        "cu_seqlens_k, block_table, and cache_seqlens must be int32"
    );

    TORCH_CHECK(k.dim() == 3 && v.dim() == 3, "k and v must be 3D tensors (Total_Lk, H_kv, 64)");
    TORCH_CHECK(k.size(2) == 64 && v.sizes() == k.sizes(), "k and v must have identical shape (Total_Lk, H_kv, 64)");
    TORCH_CHECK(
        k.stride(2) == 1 && v.stride(2) == 1 &&
        (k.stride(0) % 8 == 0) && (k.stride(1) % 8 == 0) &&
        (v.stride(0) % 8 == 0) && (v.stride(1) % 8 == 0),
        "k and v must have stride(-1) == 1 and token/head strides divisible by 8"
    );

    const int64_t Total_Lk = k.size(0);
    const int64_t H_kv     = k.size(1);
    TORCH_CHECK(H_kv > 0, "H_kv must be positive");

    constexpr int PAGE_SIZE   = 16;
    constexpr int HEAD_DIM    = 64;
    constexpr int NUM_THREADS = 128;

    TORCH_CHECK(k_cache.dim() == 4 && v_cache.sizes() == k_cache.sizes(),
                "k_cache and v_cache must have identical 4D shape (num_pages, 16, H_kv, 64)");
    TORCH_CHECK(k_cache.size(1) == PAGE_SIZE && k_cache.size(2) == H_kv && k_cache.size(3) == HEAD_DIM,
                "k_cache must have shape (num_pages, 16, H_kv, 64)");

    TORCH_CHECK(cu_seqlens_k.dim() == 1 && cu_seqlens_k.size(0) >= 1,
                "cu_seqlens_k must be a 1D tensor of length >= 1");
    const int64_t B = cu_seqlens_k.size(0) - 1;
    TORCH_CHECK(max_seqlen_k >= 0, "max_seqlen_k must be non-negative");

    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == B,
                "block_table must have shape (B, max_pages_per_seq)");
    TORCH_CHECK(cache_seqlens.dim() == 1 && cache_seqlens.size(0) == B,
                "cache_seqlens must have shape (B,)");

    // 2. Empty fast-path & kernel launch
    const c10::cuda::CUDAGuard device_guard(k.device());
    if (B == 0) {
        return;
    }
    if (max_seqlen_k == 0 || Total_Lk == 0) {
        cache_seqlens.zero_();
        return;
    }

    const int64_t max_pages_prefill = (max_seqlen_k + PAGE_SIZE - 1) / PAGE_SIZE;
    const int64_t max_pages_per_seq = block_table.size(1);
    TORCH_CHECK(max_pages_per_seq >= max_pages_prefill,
                "block_table has fewer columns than required by max_seqlen_k");

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const auto* k_ptr       = reinterpret_cast<const cutlass::half_t*>(k.data_ptr<at::Half>());
    const auto* v_ptr       = reinterpret_cast<const cutlass::half_t*>(v.data_ptr<at::Half>());
    const int32_t* cu_k_ptr = cu_seqlens_k.data_ptr<int32_t>();
    const int32_t* bt_ptr   = block_table.data_ptr<int32_t>();
    auto* k_cache_ptr       = reinterpret_cast<cutlass::half_t*>(k_cache.data_ptr<at::Half>());
    auto* v_cache_ptr       = reinterpret_cast<cutlass::half_t*>(v_cache.data_ptr<at::Half>());
    int32_t* seqlens_ptr    = cache_seqlens.data_ptr<int32_t>();

    {
        const dim3 grid(
            static_cast<unsigned int>(std::max<int64_t>(1, max_pages_prefill)),
            static_cast<unsigned int>(H_kv),
            static_cast<unsigned int>(B));
        const dim3 block(NUM_THREADS);
        kernel_prefill_to_paged_kv<PAGE_SIZE, HEAD_DIM, NUM_THREADS><<<grid, block, 0, stream>>>(
            k_ptr,
            v_ptr,
            cu_k_ptr,
            bt_ptr,
            k_cache_ptr,
            v_cache_ptr,
            seqlens_ptr,
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            static_cast<int>(H_kv),
            static_cast<int>(max_pages_per_seq));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void prefill_to_paged_kv(
    const at::Tensor& k,             // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& v,             // (Total_Lk, H_kv, 64) [FP16]
    const at::Tensor& cu_seqlens_k,  // (B + 1,) [Int32]
    at::Tensor& k_cache,             // (num_pages, 16, H_kv, 64) [FP16]
    at::Tensor& v_cache,             // (num_pages, 16, H_kv, 64) [FP16]
    const at::Tensor& block_table,   // (B, max_pages_per_seq) [Int32]
    at::Tensor& cache_seqlens,       // (B,) [Int32]
    int64_t max_seqlen_k             // scalar: int64
) {
    launch_prefill_to_paged_kv(
        k, v, cu_seqlens_k, k_cache, v_cache, block_table, cache_seqlens, max_seqlen_k
    );
}

void launch_paged_kv_append(
    const at::Tensor& new_k,         // (B, H_kv, 64) [FP16]
    const at::Tensor& new_v,         // (B, H_kv, 64) [FP16]
    at::Tensor& k_cache,             // (num_pages, 16, H_kv, 64) [FP16]
    at::Tensor& v_cache,             // (num_pages, 16, H_kv, 64) [FP16]
    const at::Tensor& block_table,   // (B, max_pages_per_seq) [Int32]
    at::Tensor& cache_seqlens        // (B,) [Int32]
) {
    // 1. Input validation
    TORCH_CHECK(new_k.is_cuda() && new_v.is_cuda(), "new_k and new_v must be CUDA tensors");
    TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda() && block_table.is_cuda() && cache_seqlens.is_cuda(),
                "k_cache, v_cache, block_table, and cache_seqlens must be CUDA tensors");
    TORCH_CHECK(
        new_v.device() == new_k.device() && k_cache.device() == new_k.device() &&
        v_cache.device() == new_k.device() && block_table.device() == new_k.device() &&
        cache_seqlens.device() == new_k.device(),
        "All tensors must reside on the same CUDA device"
    );

    TORCH_CHECK(
        k_cache.is_contiguous() && v_cache.is_contiguous() &&
        block_table.is_contiguous() && cache_seqlens.is_contiguous(),
        "k_cache, v_cache, block_table, and cache_seqlens must be contiguous"
    );
    TORCH_CHECK(
        new_k.scalar_type() == at::kHalf && new_v.scalar_type() == at::kHalf &&
        k_cache.scalar_type() == at::kHalf && v_cache.scalar_type() == at::kHalf,
        "new_k, new_v, k_cache, and v_cache must be float16"
    );
    TORCH_CHECK(block_table.scalar_type() == at::kInt && cache_seqlens.scalar_type() == at::kInt,
                "block_table and cache_seqlens must be int32");

    TORCH_CHECK(new_k.dim() == 3 && new_v.dim() == 3, "new_k and new_v must be 3D tensors (B, H_kv, 64)");
    TORCH_CHECK(new_k.size(2) == 64 && new_v.sizes() == new_k.sizes(),
                "new_k and new_v must have identical shape (B, H_kv, 64)");
    TORCH_CHECK(
        new_k.stride(2) == 1 && new_v.stride(2) == 1 &&
        (new_k.stride(0) % 8 == 0) && (new_k.stride(1) % 8 == 0) &&
        (new_v.stride(0) % 8 == 0) && (new_v.stride(1) % 8 == 0),
        "new_k and new_v must have stride(-1) == 1 and batch/head strides divisible by 8"
    );

    const int64_t B    = new_k.size(0);
    const int64_t H_kv = new_k.size(1);
    TORCH_CHECK(H_kv > 0, "H_kv must be positive");

    constexpr int PAGE_SIZE   = 16;
    constexpr int HEAD_DIM    = 64;
    constexpr int HEAD_TILE   = 16;
    constexpr int NUM_THREADS = 128;

    TORCH_CHECK(k_cache.dim() == 4 && v_cache.sizes() == k_cache.sizes(),
                "k_cache and v_cache must have identical 4D shape (num_pages, 16, H_kv, 64)");
    TORCH_CHECK(k_cache.size(1) == PAGE_SIZE && k_cache.size(2) == H_kv && k_cache.size(3) == HEAD_DIM,
                "k_cache must have shape (num_pages, 16, H_kv, 64)");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == B,
                "block_table must have shape (B, max_pages_per_seq)");
    TORCH_CHECK(cache_seqlens.dim() == 1 && cache_seqlens.size(0) == B,
                "cache_seqlens must have shape (B,)");

    // 2. Kernel launch
    const c10::cuda::CUDAGuard device_guard(new_k.device());
    if (B == 0) {
        return;
    }
    const int64_t max_pages_per_seq = block_table.size(1);
    TORCH_CHECK(max_pages_per_seq >= 1, "block_table must have at least 1 page column when B > 0");
    TORCH_CHECK(k_cache.size(0) >= 1, "k_cache must contain at least 1 allocated page when B > 0");

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const auto* nk_ptr   = reinterpret_cast<const cutlass::half_t*>(new_k.data_ptr<at::Half>());
    const auto* nv_ptr   = reinterpret_cast<const cutlass::half_t*>(new_v.data_ptr<at::Half>());
    const int32_t* bt_ptr = block_table.data_ptr<int32_t>();
    auto* k_cache_ptr    = reinterpret_cast<cutlass::half_t*>(k_cache.data_ptr<at::Half>());
    auto* v_cache_ptr    = reinterpret_cast<cutlass::half_t*>(v_cache.data_ptr<at::Half>());
    int32_t* seqlens_ptr = cache_seqlens.data_ptr<int32_t>();

    {
        const dim3 grid(static_cast<unsigned int>(B), 1U, 1U);
        const dim3 block(NUM_THREADS);
        kernel_paged_kv_append<PAGE_SIZE, HEAD_DIM, HEAD_TILE, NUM_THREADS><<<grid, block, 0, stream>>>(
            nk_ptr,
            nv_ptr,
            bt_ptr,
            k_cache_ptr,
            v_cache_ptr,
            seqlens_ptr,
            new_k.stride(0),
            new_k.stride(1),
            new_v.stride(0),
            new_v.stride(1),
            static_cast<int>(H_kv),
            static_cast<int>(max_pages_per_seq));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void paged_kv_append(
    const at::Tensor& new_k,         // (B, H_kv, 64) [FP16]
    const at::Tensor& new_v,         // (B, H_kv, 64) [FP16]
    at::Tensor& k_cache,             // (num_pages, 16, H_kv, 64) [FP16]
    at::Tensor& v_cache,             // (num_pages, 16, H_kv, 64) [FP16]
    const at::Tensor& block_table,   // (B, max_pages_per_seq) [Int32]
    at::Tensor& cache_seqlens        // (B,) [Int32]
) {
    launch_paged_kv_append(new_k, new_v, k_cache, v_cache, block_table, cache_seqlens);
}
