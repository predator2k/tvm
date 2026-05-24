/*
 * Debug: simulate AMX tile computation in pure C to isolate the bug.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline float bf16_to_f32(unsigned short v) {
  unsigned int bits = ((unsigned int)v) << 16;
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

static inline unsigned short f32_to_bf16(float v) {
  unsigned int bits;
  memcpy(&bits, &v, sizeof(bits));
  return (unsigned short)(bits >> 16);
}

static void ref_gemm_nt(int M, int N, int K, const unsigned short *A, int lda,
                        const unsigned short *B, int ldb, float *C, int ldc) {
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++)
      for (int k = 0; k < K; k++)
        C[m * ldc + n] += bf16_to_f32(A[m * lda + k]) * bf16_to_f32(B[n * ldb + k]);
}

/* Simulate exactly what the AMX code does: tile packing, dot product, tile store */
static void sim_gemm_nt(int M, int N, int K, const unsigned short *A, int lda,
                        const unsigned short *B, int ldb, float *C, int ldc) {
  const int K_TILE = 32;
  const int TILE_ROWS = 16;

  for (int m = 0; m < M; m += TILE_ROWS) {
    for (int n = 0; n < N; n += TILE_ROWS) {

      /* Accumulator tile (16x16 fp32), initially zero */
      float C_tile[16][16] = {0};

      for (int k = 0; k < K; k += K_TILE) {
        int k_len = (k + K_TILE <= K) ? K_TILE : (K - k);

        /* A_tile: 16 x 32 bf16 */
        float A_tile[16][32] = {0};
        for (int r = 0; r < TILE_ROWS && (m + r) < M; r++)
          for (int c = 0; c < k_len; c++)
            A_tile[r][c] = bf16_to_f32(A[(m + r) * lda + (k + c)]);

        /* B_tile: 16 x 32 bf16 (standard layout for TDPBF16PS) */
        float B_tile[16][32] = {0};
        for (int r = 0; r < TILE_ROWS && (n + r) < N; r++)
          for (int c = 0; c < k_len; c++)
            B_tile[r][c] = bf16_to_f32(B[(n + r) * ldb + (k + c)]);

        /* TDPBF16PS: C_tile[i][j] += Σ_k A_tile[i][k] * B_tile[j][k] */
        for (int i = 0; i < TILE_ROWS && (m + i) < M; i++)
          for (int j = 0; j < TILE_ROWS && (n + j) < N; j++)
            for (int kk = 0; kk < k_len; kk++)
              C_tile[i][j] += A_tile[i][kk] * B_tile[j][kk];
      }

      /* Store C_tile to output (like _tile_stored) */
      for (int i = 0; i < TILE_ROWS && (m + i) < M; i++)
        for (int j = 0; j < TILE_ROWS && (n + j) < N; j++)
          C[(m + i) * ldc + (n + j)] += C_tile[i][j];
    }
  }
}

int main() {
  int M = 32, N = 32, K = 32;
  int lda = 128, ldb = 128, ldc = N;

  unsigned short *A = (unsigned short *)malloc(M * lda * sizeof(unsigned short));
  unsigned short *B = (unsigned short *)malloc(N * ldb * sizeof(unsigned short));
  float *C_ref = (float *)calloc(M * N, sizeof(float));
  float *C_sim = (float *)calloc(M * N, sizeof(float));

  for (int m = 0; m < M; m++)
    for (int k = 0; k < K; k++)
      A[m * lda + k] = f32_to_bf16((float)((m * 7 + k * 3) % 11 - 5));
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++)
      B[n * ldb + k] = f32_to_bf16((float)((n * 5 + k * 2) % 7 + 1));

  ref_gemm_nt(M, N, K, A, lda, B, ldb, C_ref, ldc);
  sim_gemm_nt(M, N, K, A, lda, B, ldb, C_sim, ldc);

  float max_err = 0.0f;
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      float err = fabsf(C_ref[m * ldc + n] - C_sim[m * ldc + n]);
      if (err > max_err) max_err = err;
    }

  printf("Simulation vs Reference: max_error = %.6f\n", max_err);

  /* Print first few */
  printf("First 3x3 block:\n");
  for (int m = 0; m < 3; m++) {
    printf("  ");
    for (int n = 0; n < 3; n++)
      printf("ref=%.2f/sim=%.2f  ", C_ref[m*ldc+n], C_sim[m*ldc+n]);
    printf("\n");
  }

  free(A); free(B); free(C_ref); free(C_sim);
  return max_err > 1e-6;
}
