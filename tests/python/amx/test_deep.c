/* Deep dive: test AMX GEMM with different layouts but SAME logical data. */
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

extern void amx_bf16_gemm_nt(int M, int N, int K, const unsigned short *A, int lda,
                             const unsigned short *B, int ldb, float *C, int ldc);

static void init_amx() {
  uint64_t bitmask = 0;
  syscall(SYS_arch_prctl, 0x1022, &bitmask);
  if (!(bitmask & (1 << 18)))
    syscall(SYS_arch_prctl, 0x1023, 18);
}

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

int main() {
  init_amx();

  int M = 32, N = 32, K = 32;
  int lda_big = 128, ldb_big = 128;
  int lda_small = K, ldb_small = K;
  int ldc = N;

  /* Create data with big strides (lda=128) */
  unsigned short *A_big = malloc(M * lda_big * sizeof(unsigned short));
  unsigned short *B_big = malloc(N * ldb_big * sizeof(unsigned short));
  for (int m = 0; m < M; m++)
    for (int k = 0; k < K; k++)
      A_big[m * lda_big + k] = f32_to_bf16((float)((m * 7 + k * 3) % 11 - 5));
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++)
      B_big[n * ldb_big + k] = f32_to_bf16((float)((n * 5 + k * 2) % 7 + 1));

  /* Copy SAME data into contiguous buffers */
  unsigned short *A_small = calloc(M * lda_small, sizeof(unsigned short));
  unsigned short *B_small = calloc(N * ldb_small, sizeof(unsigned short));
  for (int m = 0; m < M; m++)
    for (int k = 0; k < K; k++)
      A_small[m * lda_small + k] = A_big[m * lda_big + k];
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++)
      B_small[n * ldb_small + k] = B_big[n * ldb_big + k];

  /* Reference */
  float *C_ref = calloc(M * ldc, sizeof(float));
  float *C_big = calloc(M * ldc, sizeof(float));
  float *C_small = calloc(M * ldc, sizeof(float));

  ref_gemm_nt(M, N, K, A_big, lda_big, B_big, ldb_big, C_ref, ldc);

  /* AMX with big strides */
  amx_bf16_gemm_nt(M, N, K, A_big, lda_big, B_big, ldb_big, C_big, ldc);

  /* AMX with small (contiguous) strides, SAME logical data */
  amx_bf16_gemm_nt(M, N, K, A_small, lda_small, B_small, ldb_small, C_small, ldc);

  /* Compare */
  float err_big = 0, err_small = 0;
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      float e_big = fabsf(C_big[m*ldc+n] - C_ref[m*ldc+n]);
      float e_small = fabsf(C_small[m*ldc+n] - C_ref[m*ldc+n]);
      if (e_big > err_big) err_big = e_big;
      if (e_small > err_small) err_small = e_small;
    }

  printf("Max error: big_stride(%d)=%.6f  small_stride(%d)=%.6f\n",
         lda_big, err_big, lda_small, err_small);

  if (err_small > 1.0)
    printf("  *** ERROR even with contiguous data! Problem is in the GEMM kernel.\n");
  if (err_big > 1.0)
    printf("  *** Error with strided data.\n");

  printf("First 3x3 C_ref:\n");
  for (int m = 0; m < 3; m++) {
    printf("  ");
    for (int n = 0; n < 3; n++) printf("%.1f  ", C_ref[m*ldc+n]);
    printf("\n");
  }
  printf("First 3x3 C_big (stride=%d):\n", lda_big);
  for (int m = 0; m < 3; m++) {
    printf("  ");
    for (int n = 0; n < 3; n++) printf("%.1f  ", C_big[m*ldc+n]);
    printf("\n");
  }
  printf("First 3x3 C_small (stride=%d):\n", lda_small);
  for (int m = 0; m < 3; m++) {
    printf("  ");
    for (int n = 0; n < 3; n++) printf("%.1f  ", C_small[m*ldc+n]);
    printf("\n");
  }

  free(A_big); free(B_big); free(A_small); free(B_small);
  free(C_ref); free(C_big); free(C_small);
  return (err_big > 1.0) || (err_small > 1.0);
}
