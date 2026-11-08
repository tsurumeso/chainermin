// ============================================================================
// Scaled Dot-Product Attention (SDPA) 用 Fused Scale + Causal Mask + Softmax カーネル
// ============================================================================
//
// 【背景・目的】
// Attention の計算式:
//   P = softmax( (Q @ K^T) * scale + causal_mask )
//   O = P @ V
//
// 行列積 (Q @ K^T や P @ V) は cuBLAS の Tensor Core で非常に高速に計算できますが、
// その間に挟まる「スケール倍・Causal Mask 適用・Softmax」を個別の演算として実行すると、
// 巨大な (B, H, S_q, S_kv) 行列に対して Forward / Backward 合計で 20回以上の
// VRAM (Global Memory) 読み書きが発生し、メモリ帯域がボトルネックになります。
//
// このカーネルでは、1つの CUDA ブロック (128スレッド = 4 Warp) に Attention 行列の
// 「1行 (長さ S_kv)」を割り当て、以下の処理を 1回のカーネル起動でインプレースに行います:
//   1. スケール倍 (scores * scale)
//   2. Causal Mask (未来のトークン位置 col > row_q をマスク)
//   3. Softmax の順伝播 / 逆伝播 (FP32 精度での Warp 並列リダクション)
//
// さらに S_kv <= 1024 の場合は、各スレッドが担当する要素をすべて GPU のレジスタに
// キャッシュするため、VRAM からの読み込みが「1回のみ」で完結します。
// ============================================================================

#include <cuda_fp16.h>

// ----------------------------------------------------------------------------
// 型変換ヘルパー (FP16 / FP32 共通化のため)
// 入出力が FP16 (__half) の場合でも、指数計算 (exp) や総和 (sum) の丸め誤差・
// オーバーフローを防ぐため、カーネル内部の計算はすべて FP32 (float) で行います。
// ----------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float to_float(T x) {
    return static_cast<float>(x);
}
template <>
__device__ __forceinline__ float to_float<__half>(__half x) {
    return __half2float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x) {
    return static_cast<T>(x);
}
template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
    return __float2half(x);
}

// ----------------------------------------------------------------------------
// ブロック内での最大値リダクション (Block-wide Max Reduction)
//
// 1ブロック (BLOCK_SIZE スレッド) が持つそれぞれの値 `val` の中から、
// ブロック全体の最大値を求めて全スレッドに返します。
//
// 【2段階リダクションの仕組み】
// 1. Warp 内リダクション:
//    1 Warp (32スレッド) 内では、Shared Memory を使わずに `__shfl_xor_sync` 命令で
//    スレッド間のレジスタを直接交換できます (offset = 16 -> 8 -> 4 -> 2 -> 1 の5ステップ)。
// 2. Warp 間リダクション:
//    各 Warp の先頭スレッド (lane == 0) が Warp 内の最大値を Shared Memory に書き込み、
//    最初の Warp (wid == 0) がそれらを集約して最終結果を `shared[0]` に格納します。
// ----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
    const int lane = threadIdx.x & 31;  // Warp 内のスレッド番号 (0 ~ 31)
    const int wid = threadIdx.x >> 5;   // ブロック内の Warp 番号 (0 ~ BLOCK_SIZE/32 - 1)

    // 1. Warp 内でトーナメント方式に最大値を集約
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }

    // 各 Warp の代表 (lane == 0) が Shared Memory に書き込む
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // 2. 先頭 Warp (wid == 0) が全 Warp の代表値を読み込んでさらに集約
    val = (threadIdx.x < (BLOCK_SIZE >> 5)) ? shared[lane] : -1e38f;
    if (wid == 0) {
        #pragma unroll
        for (int offset = (BLOCK_SIZE >> 6); offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
        }
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();

    // ブロック内の全スレッドが同じ最大値を受け取る
    return shared[0];
}

// ----------------------------------------------------------------------------
// ブロック内での総和リダクション (Block-wide Sum Reduction)
// `block_reduce_max` と同じ手順で、ブロック全体の総和 (sum) を計算します。
// ----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    // 1. Warp 内で総和を集約
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // 2. 先頭 Warp で全 Warp の総和を集約
    val = (threadIdx.x < (BLOCK_SIZE >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int offset = (BLOCK_SIZE >> 6); offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();

    return shared[0];
}

// ============================================================================
// Forward カーネル実装: scores をインプレースで Softmax 確率 P に変換
//
// 計算式:
//   x_j = scores[row, j] * scale   (ただし Causal Mask 範囲外 j >= limit は -inf 相当)
//   m   = max_{j < limit} (x_j)
//   P_j = exp(x_j - m) / sum_{l < limit} exp(x_l - m)   (j < limit)
//   P_j = 0                                             (j >= limit)
//
// テンプレート引数:
//   T         : データ型 (__half または float)
//   BLOCK_SIZE: 1ブロックあたりのスレッド数 (128)
//   MAX_ITEMS : 1スレッドがレジスタに保持できる最大要素数 (8 -> 128 * 8 = 1024要素まで対応)
// ============================================================================
template <typename T, int BLOCK_SIZE, int MAX_ITEMS>
__device__ void sdpa_softmax_fwd_impl(
    T* __restrict__ scores,
    const int S_q,
    const int S_kv,
    const float scale,
    const int is_causal
) {
    // Warp 間通信用の Shared Memory (128スレッドなら 128/32 = 4 float 分のみ)
    __shared__ float shared[BLOCK_SIZE >> 5];

    // 1ブロック = Attention 行列の 1行を担当
    const int row_global = blockIdx.x;   // 全体での行インデックス (0 ~ B*H*S_q - 1)
    const int row_q = row_global % S_q;  // クエリ系列内の位置 (0 ~ S_q - 1)

    // Causal Mask の有効列数 `limit`:
    // クエリ位置が `row_q` のとき、参照できるキー位置 `col` は `0 <= col <= row_q` のみ。
    // `col >= limit` の未来のトークンは確率が必ず 0 になるため、VRAM からの読み込み自体を省略できます。
    const int limit = is_causal ? ((row_q + 1 < S_kv) ? (row_q + 1) : S_kv) : S_kv;

    // このブロックが担当する行の先頭ポインタ
    T* row_ptr = scores + static_cast<long long>(row_global) * S_kv;

    if (S_kv <= BLOCK_SIZE * MAX_ITEMS) {
        // --------------------------------------------------------------------
        // 【高速パス】S_kv <= 1024 の場合:
        // 各スレッドが最大 MAX_ITEMS (8) 個の要素をレジスタ配列 `reg` に保持します。
        // VRAM からの Read は最初の1回だけで済み、以降の計算はすべてレジスタ上で完結します。
        // --------------------------------------------------------------------
        float reg[MAX_ITEMS];
        float thread_max = -1e38f;

        // Step 1: VRAM から読み出しつつ scale を掛け、スレッド内の最大値を求める
        #pragma unroll
        for (int k = 0; k < MAX_ITEMS; ++k) {
            int col = threadIdx.x + k * BLOCK_SIZE;
            if (col < limit) {
                float v = to_float<T>(row_ptr[col]) * scale;
                reg[k] = v;
                thread_max = fmaxf(thread_max, v);
            }
        }
        // ブロック全体で行の最大値 `row_max` を共有
        const float row_max = block_reduce_max<BLOCK_SIZE>(thread_max, shared);

        // Step 2: オーバーフロー防止のため `row_max` を引いてから exp を計算し、総和を求める
        float thread_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < MAX_ITEMS; ++k) {
            int col = threadIdx.x + k * BLOCK_SIZE;
            if (col < limit) {
                float e = __expf(reg[k] - row_max);
                reg[k] = e;
                thread_sum += e;
            }
        }
        // ブロック全体で exp の総和 `sum_val` を共有し、逆数を計算
        const float sum_val = block_reduce_sum<BLOCK_SIZE>(thread_sum, shared);
        const float inv_sum = (sum_val > 0.0f) ? (1.0f / sum_val) : 0.0f;

        // Step 3: 正規化した確率を VRAM に書き戻す (マスク領域 col >= limit には 0 を書き込む)
        const T zero = from_float<T>(0.0f);
        #pragma unroll
        for (int k = 0; k < MAX_ITEMS; ++k) {
            int col = threadIdx.x + k * BLOCK_SIZE;
            if (col < limit) {
                row_ptr[col] = from_float<T>(reg[k] * inv_sum);
            } else if (col < S_kv) {
                row_ptr[col] = zero;
            }
        }
    } else {
        // --------------------------------------------------------------------
        // 【汎用パス】S_kv > 1024 の場合:
        // レジスタ配列に収まらない長い系列長では、VRAM を走査して計算します。
        // --------------------------------------------------------------------
        float thread_max = -1e38f;
        for (int col = threadIdx.x; col < limit; col += BLOCK_SIZE) {
            float v = to_float<T>(row_ptr[col]) * scale;
            thread_max = fmaxf(thread_max, v);
        }
        const float row_max = block_reduce_max<BLOCK_SIZE>(thread_max, shared);

        float thread_sum = 0.0f;
        for (int col = threadIdx.x; col < limit; col += BLOCK_SIZE) {
            float v = to_float<T>(row_ptr[col]) * scale;
            thread_sum += __expf(v - row_max);
        }
        const float sum_val = block_reduce_sum<BLOCK_SIZE>(thread_sum, shared);
        const float inv_sum = (sum_val > 0.0f) ? (1.0f / sum_val) : 0.0f;

        const T zero = from_float<T>(0.0f);
        for (int col = threadIdx.x; col < S_kv; col += BLOCK_SIZE) {
            if (col < limit) {
                float v = to_float<T>(row_ptr[col]) * scale;
                row_ptr[col] = from_float<T>(__expf(v - row_max) * inv_sum);
            } else {
                row_ptr[col] = zero;
            }
        }
    }
}

// ============================================================================
// Backward カーネル実装: Softmax 出力 P と勾配 dP から、スコア勾配 dS をインプレース計算
//
// 【数式の導出】
// Softmax の逆伝播:
//   dS_raw[col] = P[col] * (dP[col] - sum_j (P[j] * dP[j]))
// これに Forward 時のスケール倍 (* scale) と Causal Mask の逆伝播を組み合わせると:
//   row_dot = sum_{j < limit} (P[j] * dP[j])
//   dS[col] = P[col] * (dP[col] - row_dot) * scale   (col < limit)
//   dS[col] = 0                                      (col >= limit)
//
// 入力の `dp` バッファをそのまま出力 `dS` で上書き (インプレース更新) します。
// ============================================================================
template <typename T, int BLOCK_SIZE, int MAX_ITEMS>
__device__ void sdpa_softmax_bwd_impl(
    const T* __restrict__ probs,
    T* __restrict__ dp,
    const int S_q,
    const int S_kv,
    const float scale,
    const int is_causal
) {
    __shared__ float shared[BLOCK_SIZE >> 5];
    const int row_global = blockIdx.x;
    const int row_q = row_global % S_q;
    const int limit = is_causal ? ((row_q + 1 < S_kv) ? (row_q + 1) : S_kv) : S_kv;

    const long long offset = static_cast<long long>(row_global) * S_kv;
    const T* p_ptr = probs + offset;
    T* dp_ptr = dp + offset;

    if (S_kv <= BLOCK_SIZE * MAX_ITEMS) {
        // --------------------------------------------------------------------
        // 【高速パス】S_kv <= 1024 の場合:
        // P と dP をレジスタに1回だけ読み込み、内積 sum(P * dP) を求めて dS を書き戻します。
        // --------------------------------------------------------------------
        float reg_p[MAX_ITEMS];
        float reg_dp[MAX_ITEMS];
        float thread_dot = 0.0f;

        // Step 1: P と dP をレジスタに読み込みつつ、内積 sum(P * dP) の部分和を計算
        #pragma unroll
        for (int k = 0; k < MAX_ITEMS; ++k) {
            int col = threadIdx.x + k * BLOCK_SIZE;
            if (col < limit) {
                float p_val = to_float<T>(p_ptr[col]);
                float dp_val = to_float<T>(dp_ptr[col]);
                reg_p[k] = p_val;
                reg_dp[k] = dp_val;
                thread_dot += p_val * dp_val;
            }
        }
        // ブロック全体で行の内積 `row_dot = sum_j(P_j * dP_j)` を共有
        const float row_dot = block_reduce_sum<BLOCK_SIZE>(thread_dot, shared);

        // Step 2: `P * (dP - row_dot) * scale` を計算し、`dp` バッファにインプレースで書き戻す
        const T zero = from_float<T>(0.0f);
        #pragma unroll
        for (int k = 0; k < MAX_ITEMS; ++k) {
            int col = threadIdx.x + k * BLOCK_SIZE;
            if (col < limit) {
                dp_ptr[col] = from_float<T>(reg_p[k] * (reg_dp[k] - row_dot) * scale);
            } else if (col < S_kv) {
                dp_ptr[col] = zero;
            }
        }
    } else {
        // --------------------------------------------------------------------
        // 【汎用パス】S_kv > 1024 の場合
        // --------------------------------------------------------------------
        float thread_dot = 0.0f;
        for (int col = threadIdx.x; col < limit; col += BLOCK_SIZE) {
            thread_dot += to_float<T>(p_ptr[col]) * to_float<T>(dp_ptr[col]);
        }
        const float row_dot = block_reduce_sum<BLOCK_SIZE>(thread_dot, shared);

        const T zero = from_float<T>(0.0f);
        for (int col = threadIdx.x; col < S_kv; col += BLOCK_SIZE) {
            if (col < limit) {
                float p_val = to_float<T>(p_ptr[col]);
                float dp_val = to_float<T>(dp_ptr[col]);
                dp_ptr[col] = from_float<T>(p_val * (dp_val - row_dot) * scale);
            } else {
                dp_ptr[col] = zero;
            }
        }
    }
}

// ============================================================================
// CuPy (RawModule) から呼び出すための C リンケージ・エントリポイント
// ============================================================================
extern "C" __global__ void sdpa_softmax_fwd_fp16(
    __half* __restrict__ scores, int S_q, int S_kv, float scale, int is_causal
) {
    sdpa_softmax_fwd_impl<__half, 128, 8>(scores, S_q, S_kv, scale, is_causal);
}

extern "C" __global__ void sdpa_softmax_fwd_fp32(
    float* __restrict__ scores, int S_q, int S_kv, float scale, int is_causal
) {
    sdpa_softmax_fwd_impl<float, 128, 8>(scores, S_q, S_kv, scale, is_causal);
}

extern "C" __global__ void sdpa_softmax_bwd_fp16(
    const __half* __restrict__ probs, __half* __restrict__ dp, int S_q, int S_kv, float scale, int is_causal
) {
    sdpa_softmax_bwd_impl<__half, 128, 8>(probs, dp, S_q, S_kv, scale, is_causal);
}

extern "C" __global__ void sdpa_softmax_bwd_fp32(
    const float* __restrict__ probs, float* __restrict__ dp, int S_q, int S_kv, float scale, int is_causal
) {
    sdpa_softmax_bwd_impl<float, 128, 8>(probs, dp, S_q, S_kv, scale, is_causal);
}
