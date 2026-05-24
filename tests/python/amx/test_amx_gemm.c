/*
 * Standalone test for AMX BF16 GEMM correctness.
 * Compile: gcc -O0 -march=sapphirerapids -o test_amx_gemm test_amx_gemm.c amx_helpers.c -lm
 */
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

static void amx_init() {
  uint64_t bitmask = 0;
  int64_t status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
  if (status != 0) {
    fprintf(stderr, "AMX init: ARCH_GET_XCOMP_PERM failed: %s\n", strerror(errno));
    exit(1);
  }
  if (bitmask & XFEATURE_MASK_XTILEDATA) {
    printf("AMX already enabled (bitmask=0x%lx)\n", bitmask);
    return;
  }
  status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (status != 0) {
    fprintf(stderr, "AMX init: ARCH_REQ_XCOMP_PERM failed: %s\n", strerror(errno));
    exit(1);
  }
  printf("AMX enabled successfully\n");
}

extern void amx_bf16_gemm_nt(int M, int N, int K, const unsigned short *A,
                             int lda, const unsigned short *B, int ldb, float *C,
                             int ldc);

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
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = C[m * ldc + n];
      for (int k = 0; k < K; k++) {
        sum += bf16_to_f32(A[m * lda + k]) * bf16_to_f32(B[n * ldb + k]);
      }
      C[m * ldc + n] = sum;
    }
  }
}

static int test_small() {
  /* Test: 32x32 x 32x16 = 32x16, with K=32 */
  int M = 32, N = 16, K = 32;
  int lda = K, ldb = K, ldc = N;

  unsigned short *A = (unsigned short *)malloc(M * K * sizeof(unsigned short));
  unsigned short *B = (unsigned short *)malloc(N * K * sizeof(unsigned short));
  float *C_amx = (float *)calloc(M * N, sizeof(float));
  float *C_ref = (float *)calloc(M * N, sizeof(float));

  /* Fill A and B with known small values */
  for (int i = 0; i < M * K; i++)
    A[i] = f32_to_bf16((float)(i % 7 - 3));   /* values -3..3 */
  for (int i = 0; i < N * K; i++)
    B[i] = f32_to_bf16((float)(i % 5 + 1));   /* values 1..5 */

  /* Reference */
  memcpy(C_ref, C_amx, M * N * sizeof(float));
  ref_gemm_nt(M, N, K, A, lda, B, ldb, C_ref, ldc);

  /* AMX */
  amx_bf16_gemm_nt(M, N, K, A, lda, B, ldb, C_amx, ldc);

  /* Compare */
  float max_err = 0.0f;
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float err = fabsf(C_amx[m * ldc + n] - C_ref[m * ldc + n]);
      if (err > max_err) max_err = err;
    }
  }

  printf("Small test (M=%d,N=%d,K=%d): max_error = %.6f\n", M, N, K, max_err);

  if (max_err > 1.0f) {
    printf("  FAIL: large error detected, dumping first few elements:\n");
    for (int m = 0; m < 3; m++) {
      for (int n = 0; n < 3; n++) {
        printf("  C[%d][%d]: ref=%.4f  amx=%.4f  diff=%.4f\n",
               m, n, C_ref[m * ldc + n], C_amx[m * ldc + n],
               C_amx[m * ldc + n] - C_ref[m * ldc + n]);
      }
    }
  }

  free(A); free(B); free(C_amx); free(C_ref);
  return max_err > 1.0f;
}

static int test_packed() {
  /* Test with lda != K (packed layout) — like the MHA second matmul:
   * C = P @ V^T where P is 128x128 and V^T is 32x128.
   * Use a 32x32 subset for fast testing. */
  int M = 32, N = 32, K = 32;
  int lda = 128, ldb = 128, ldc = N;

  unsigned short *A = (unsigned short *)malloc(M * lda * sizeof(unsigned short));
  unsigned short *B = (unsigned short *)malloc(N * ldb * sizeof(unsigned short));
  float *C_amx = (float *)calloc(M * N, sizeof(float));
  float *C_ref = (float *)calloc(M * N, sizeof(float));

  for (int m = 0; m < M; m++)
    for (int k = 0; k < K; k++)
      A[m * lda + k] = f32_to_bf16((float)((m * 7 + k * 3) % 11 - 5));
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++)
      B[n * ldb + k] = f32_to_bf16((float)((n * 5 + k * 2) % 7 + 1));

  memcpy(C_ref, C_amx, M * N * sizeof(float));
  ref_gemm_nt(M, N, K, A, lda, B, ldb, C_ref, ldc);
  amx_bf16_gemm_nt(M, N, K, A, lda, B, ldb, C_amx, ldc);

  float max_err = 0.0f;
  int max_m = 0, max_n = 0;
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      float err = fabsf(C_amx[m * ldc + n] - C_ref[m * ldc + n]);
      if (err > max_err) { max_err = err; max_m = m; max_n = n; }
    }

  printf("Packed test (M=%d,N=%d,K=%d,lda=%d,ldb=%d): max_error = %.6f at [%d][%d]\n",
         M, N, K, lda, ldb, max_err, max_m, max_n);
  printf("  C_ref[%d][%d]=%.4f  C_amx[%d][%d]=%.4f\n",
         max_m, max_n, C_ref[max_m * ldc + max_n],
         max_m, max_n, C_amx[max_m * ldc + max_n]);
  /* Print first few V_ref vs V_amx */
  printf("  First 3x3 block:\n");
  for (int m = 0; m < 3; m++) {
    printf("    ");
    for (int n = 0; n < 3; n++)
      printf("ref=%.2f/amx=%.2f  ", C_ref[m*ldc+n], C_amx[m*ldc+n]);
    printf("\n");
  }
  /* Print raw A and B values used in tile (m=0,n=0) */
  printf("  A[0][0..3]: %.4f %.4f %.4f %.4f\n",
         bf16_to_f32(A[0*lda+0]), bf16_to_f32(A[0*lda+1]),
         bf16_to_f32(A[0*lda+2]), bf16_to_f32(A[0*lda+3]));
  printf("  B[0][0..3]: %.4f %.4f %.4f %.4f\n",
         bf16_to_f32(B[0*ldb+0]), bf16_to_f32(B[0*ldb+1]),
         bf16_to_f32(B[0*ldb+2]), bf16_to_f32(B[0*ldb+3]));

  free(A); free(B); free(C_amx); free(C_ref);
  return max_err > 1.0f;
}

int main() {
  amx_init();
  int fail = 0;
  fail |= test_small();
  fail |= test_packed();
  printf("\n%s\n", fail ? "TESTS FAILED" : "All tests passed!");
  return fail;
}
