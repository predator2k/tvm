/*
 * AMX BF16 GEMM — B packed in N_BLOCK-sized panels for cache efficiency.
 * Each B panel: [K/2 rows][N_BLOCK*2 cols], row stride = N_BLOCK*2 ~ 1KB.
 *
 * This keeps consecutive K-pair rows close together in memory, avoiding
 * TLB/cache misses when TDPBF16PS loads 16 consecutive K-pair rows.
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void cfg_tiles(int rows, int colsb) {
  unsigned char cfg[64] __attribute__((aligned(64))) = {0};
  cfg[0] = 1;
  for (int i = 0; i < 8; i++) {
    *(uint16_t *)(cfg + 16 + 2 * i) = (uint16_t)colsb;
    cfg[48 + i] = (uint8_t)rows;
  }
  _tile_loadconfig(cfg);
}

/*
 * GEMM kernel using pre-packed B panels.
 * A: [M][K] row-major bf16, lda = K
 * B_packed: [K/2][N_packed] where N_packed is the panel width (N_BLOCK*2)
 * C: [M][N] row-major fp32, ldc = N
 */
void amx_bf16_gemm_prepacked(
    int M, int N, int K,
    const unsigned short * __restrict A, int lda,
    const unsigned short * __restrict B_packed, int N_packed,
    float * __restrict C, int ldc) {
  cfg_tiles(16, 64);

  const int M_TILE = 16, N_TILE = 16, K_TILE_PAIRS = 16;
  int K_pairs = K / 2;
  int a_stride = lda * 2;          /* bytes between rows in A */
  int b_stride = N_packed * 2;     /* bytes between K-pair rows in B_packed */

  /* GEMM over full M × N, with K pre-packed */
  for (int m = 0; m < M; m += M_TILE) {
    for (int n = 0; n < N; n += N_TILE) {
      _tile_zero(4);  /* accumulator in tile 4 */

      /* Double-buffered K loop */
      int kk = 0;

      /* Pre-load A into tile 0, B into tile 1 for first K_TILE_PAIRS */
      if (kk < K_pairs) {
        _tile_stream_loadd(0, A + (size_t)m * lda + kk * 2, a_stride);
        _tile_stream_loadd(1, B_packed + (size_t)kk * N_packed + n * 2, b_stride);
        kk += K_TILE_PAIRS;
      }
      if (kk < K_pairs) {
        _tile_stream_loadd(2, A + (size_t)m * lda + kk * 2, a_stride);
        _tile_stream_loadd(3, B_packed + (size_t)kk * N_packed + n * 2, b_stride);
        kk += K_TILE_PAIRS;
      }

      /* Main loop: interleaved compute on {0,1}, {2,3} while loading next */
      for (; kk + K_TILE_PAIRS <= K_pairs; kk += K_TILE_PAIRS) {
        _tile_dpbf16ps(4, 0, 1);  /* compute with tiles 0,1 */
        /* Load next into 0,1 */
        _tile_stream_loadd(0, A + (size_t)m * lda + kk * 2, a_stride);
        _tile_stream_loadd(1, B_packed + (size_t)kk * N_packed + n * 2, b_stride);
        kk += K_TILE_PAIRS;

        if (kk < K_pairs) {
          _tile_dpbf16ps(4, 2, 3);  /* compute with tiles 2,3 */
          /* Load next into 2,3 */
          _tile_stream_loadd(2, A + (size_t)m * lda + kk * 2, a_stride);
          _tile_stream_loadd(3, B_packed + (size_t)kk * N_packed + n * 2, b_stride);
        }
      }

      /* Drain remaining */
      _tile_dpbf16ps(4, 0, 1);
      if (K_pairs > K_TILE_PAIRS)
        _tile_dpbf16ps(4, 2, 3);

      _tile_stored(4, C + (size_t)m * ldc + n, ldc * 4);
    }
  }
}

/*
 * Convenience: GEMM that internally pre-packs B in N_BLOCK panels.
 */
void amx_bf16_gemm_large(int M, int N, int K,
                         const unsigned short * __restrict A, int lda,
                         const unsigned short * __restrict B, int ldb,
                         float * __restrict C, int ldc) {
  const int N_BLOCK = 512;  /* process N in 512-wide panels */
  int K_pairs = K / 2;
  int N_packed = N_BLOCK * 2;  /* each B panel row has N_BLOCK pairs */

  for (int n0 = 0; n0 < N; n0 += N_BLOCK) {
    int n_len = (n0 + N_BLOCK <= N) ? N_BLOCK : (N - n0);

    /* Pack B panel for this N-block: [K_pairs][n_len*2] */
    unsigned short *B_panel = (unsigned short *)aligned_alloc(64,
        (size_t)K_pairs * n_len * 2 * sizeof(unsigned short));

    for (int kk = 0; kk < K_pairs; kk++) {
      unsigned short *row = B_panel + (size_t)kk * n_len * 2;
      int k_even = 2 * kk, k_odd = 2 * kk + 1;
      for (int nn = 0; nn < n_len; nn++) {
        int n_idx = n0 + nn;
        row[2 * nn]     = B[n_idx * ldb + k_even];
        row[2 * nn + 1] = B[n_idx * ldb + k_odd];
      }
    }

    /* Run GEMM on this panel */
    amx_bf16_gemm_prepacked(M, n_len, K, A, lda,
                            B_panel, n_len * 2,
                            C + n0, ldc);
    free(B_panel);
  }
}

/* Multi-threaded: each thread packs B independently */
void amx_bf16_gemm_parallel(int M, int N, int K,
                            const unsigned short * __restrict A, int lda,
                            const unsigned short * __restrict B, int ldb,
                            float * __restrict C, int ldc,
                            int num_threads) {
#ifdef _OPENMP
  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int m0 = 0; m0 < M; m0 += 2048) {
    int mL = (m0 + 2048 <= M) ? 2048 : (M - m0);
    amx_bf16_gemm_large(mL, N, K, A + m0 * lda, lda, B, ldb, C + m0 * ldc, ldc);
  }
#else
  (void)num_threads;
  amx_bf16_gemm_large(M, N, K, A, lda, B, ldb, C, ldc);
#endif
}

void fp32_to_bf16_convert(const float *s, unsigned short *d, int n) {
  for (int i = 0; i < n; i++) {
    unsigned int b; memcpy(&b, &s[i], 4); d[i] = (unsigned short)(b >> 16);
  }
}
void bf16_to_fp32_convert(const unsigned short *s, float *d, int n) {
  for (int i = 0; i < n; i++) {
    unsigned int b = ((unsigned int)s[i]) << 16; memcpy(&d[i], &b, 4);
  }
}
