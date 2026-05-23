/*
 * AMX helper functions for BF16 matrix multiply (GCC 11 compatible).
 * Compile: gcc -shared -fPIC -O3 -march=sapphirerapids -o libamx_helpers.so amx_helpers.c
 *
 * GCC 11 AMX intrinsics use tile register numbers (0-7), not tile types.
 *
 * Tile config layout (64 bytes, palette 1):
 *   byte 0:    palette_id = 1
 *   byte 1:    start_row = 0
 *   bytes 2-15: reserved
 *   bytes 16-31: colsb[0..7] (uint16) -- bytes per row per tile
 *   bytes 32-47: reserved
 *   bytes 48-55: rows[0..7] (uint8) -- rows per tile
 *   bytes 56-63: reserved
 *
 * TDPBF16PS pseudocode (from Intel SDM):
 *   C[m][n] += Σ_{k=0}^{15} (
 *       A[m][2k+0] * B_tile.row[k].bf16[2n+0] +
 *       A[m][2k+1] * B_tile.row[k].bf16[2n+1]
 *   )
 * A (src1) is m×k: rows = M, cols = K (standard row-major).
 * B (src2) is k×n: rows = K, cols = N (K as row dimension!).
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void configure_tiles(int rows, int colsb) {
  unsigned char cfg[64] = {0};
  cfg[0] = 1; /* palette_id */
  cfg[1] = 0; /* start_row */
  for (int i = 0; i < 8; i++) {
    *(uint16_t *)(cfg + 16 + 2 * i) = (uint16_t)colsb;
    cfg[48 + i] = (uint8_t)rows;
  }
  _tile_loadconfig(cfg);
}

/*
 * C[M][N] += A[M][K] * B[N][K]   (A,B: bf16 row-major; C: fp32 row-major)
 *
 * Processes K in chunks of 32 (one AMX tile row = 64 bytes).
 * Output blocked 16x16.
 */
void amx_bf16_gemm_nt(int M, int N, int K, const unsigned short *A, int lda,
                      const unsigned short *B, int ldb, float *C, int ldc) {
  const int K_TILE = 32;
  configure_tiles(16, 64);

  for (int m = 0; m < M; m += 16) {
    for (int n = 0; n < N; n += 16) {
      _tile_zero(2); /* accumulator in tile register 2 */

      for (int k = 0; k < K; k += K_TILE) {
        int k_len = (k + K_TILE <= K) ? K_TILE : (K - k);

        /*
         * A_tile (tile 0): standard layout, 16 rows x 32 bf16 elements.
         * A_tile[row][col] = A[m+row][k+col]
         */
        unsigned short a_packed[16 * 32];
        memset(a_packed, 0, sizeof(a_packed));
        for (int r = 0; r < 16 && (m + r) < M; r++) {
          for (int c = 0; c < k_len; c++) {
            a_packed[r * 32 + c] = A[(m + r) * lda + (k + c)];
          }
        }
        _tile_stream_loadd(0, a_packed, 64);

        /*
         * B_tile (tile 1, src2): K-pairs as ROWS, N-pairs as COLUMNS.
         *
         * TDPBF16PS pseudocode:
         *   C[m][n] += Σ_{k=0}^{15} (
         *       A[m][2k+0] * B_tile.row[k].bf16[2n+0] +
         *       A[m][2k+1] * B_tile.row[k].bf16[2n+1]
         *   )
         *
         * Where B_tile is a k×n matrix (K rows × N cols).
         * So B_tile.row[k].bf16[2n+0] = B_orig[n][k*2+0]
         *    B_tile.row[k].bf16[2n+1] = B_orig[n][k*2+1]
         */
        unsigned short b_packed[16 * 32];
        memset(b_packed, 0, sizeof(b_packed));
        for (int kk = 0; kk < k_len / 2; kk++) {
          for (int nn = 0; nn < 16 && (n + nn) < N; nn++) {
            b_packed[kk * 32 + 2 * nn] = B[(n + nn) * ldb + (k + 2 * kk)];
            b_packed[kk * 32 + 2 * nn + 1] = B[(n + nn) * ldb + (k + 2 * kk + 1)];
          }
        }
        _tile_stream_loadd(1, b_packed, 64);

        /* C_tile[16x16 fp32] += A_tile @ B_tile^T */
        _tile_dpbf16ps(2, 0, 1);
      }

      /* Store C[m:m+16, n:n+16] */
      _tile_stored(2, C + m * ldc + n, ldc * 4);
    }
  }
}

void fp32_to_bf16(const float *src, unsigned short *dst, int n) {
  for (int i = 0; i < n; i++) {
    unsigned int *fp = (unsigned int *)&src[i];
    dst[i] = (unsigned short)(*fp >> 16);
  }
}

void softmax_fp32(float *data, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    float *row = data + i * cols;
    float mx = row[0];
    for (int j = 1; j < cols; j++)
      if (row[j] > mx) mx = row[j];
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
      row[j] = expf(row[j] - mx);
      sum += row[j];
    }
    for (int j = 0; j < cols; j++) {
      row[j] /= sum;
    }
  }
}
