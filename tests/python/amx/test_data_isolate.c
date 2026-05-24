/*
 * Test: feed the problematic data through the MINIMAL tile operations,
 * bypassing the amx_bf16_gemm_nt function entirely.
 */
#include <errno.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

static void init_amx() {
  uint64_t bitmask = 0;
  syscall(SYS_arch_prctl, 0x1022, &bitmask);
  if (!(bitmask & (1 << 18)))
    syscall(SYS_arch_prctl, 0x1023, 18);
}

static void cfg_tiles(int rows, int colsb) {
  unsigned char cfg[64] = {0};
  cfg[0] = 1;
  for (int i = 0; i < 8; i++) {
    *(uint16_t *)(cfg + 16 + 2 * i) = (uint16_t)colsb;
    cfg[48 + i] = (uint8_t)rows;
  }
  _tile_loadconfig(cfg);
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

int main() {
  init_amx();
  cfg_tiles(16, 64);

  int M = 32, N = 32, K = 32;

  /* Test 1: the "working" data from the small test */
  printf("=== Test 1: Working data (flat filled) ===\n");
  {
    unsigned short A[32 * 32], B[32 * 32];
    for (int i = 0; i < 32 * 32; i++) {
      A[i] = f32_to_bf16((float)(i % 7 - 3));
      B[i] = f32_to_bf16((float)(i % 5 + 1));
    }

    /* Reference */
    float C_ref[16][16];
    for (int i = 0; i < 16; i++)
      for (int j = 0; j < 16; j++) {
        float s = 0;
        for (int k = 0; k < 32; k++)
          s += bf16_to_f32(A[i*32+k]) * bf16_to_f32(B[j*32+k]);
        C_ref[i][j] = s;
      }

    /* Pack into 64-byte-stride buffers */
    unsigned short Ap[16*32], Bp[16*32];
    for (int i = 0; i < 16; i++) {
      memcpy(&Ap[i*32], &A[i*32], 64);
      memcpy(&Bp[i*32], &B[i*32], 64);
    }

    _tile_zero(2);
    _tile_stream_loadd(0, Ap, 64);
    _tile_stream_loadd(1, Bp, 64);
    _tile_dpbf16ps(2, 0, 1);

    float C_amx[16][16];
    _tile_stored(2, C_amx, 64);

    float err = 0;
    for (int i = 0; i < 16; i++)
      for (int j = 0; j < 16; j++)
        err = fmaxf(err, fabsf(C_amx[i][j] - C_ref[i][j]));
    printf("  Max error: %.6f\n", err);

    /* Print first few */
    printf("  Ref[0][0..2]: %.1f %.1f %.1f\n", C_ref[0][0], C_ref[0][1], C_ref[0][2]);
    printf("  AMX[0][0..2]: %.1f %.1f %.1f\n", C_amx[0][0], C_amx[0][1], C_amx[0][2]);
  }

  /* Test 2: the "failing" data from the packed test */
  printf("\n=== Test 2: Failing data (row-dependent) ===\n");
  {
    unsigned short A[32 * 128], B[32 * 128];
    int lda = 128, ldb = 128;
    for (int m = 0; m < 32; m++)
      for (int k = 0; k < 32; k++)
        A[m * lda + k] = f32_to_bf16((float)((m * 7 + k * 3) % 11 - 5));
    for (int n = 0; n < 32; n++)
      for (int k = 0; k < 32; k++)
        B[n * ldb + k] = f32_to_bf16((float)((n * 5 + k * 2) % 7 + 1));

    /* Reference */
    float C_ref[16][16];
    for (int i = 0; i < 16; i++)
      for (int j = 0; j < 16; j++) {
        float s = 0;
        for (int k = 0; k < 32; k++)
          s += bf16_to_f32(A[i*lda+k]) * bf16_to_f32(B[j*ldb+k]);
        C_ref[i][j] = s;
      }

    /* Pack into 64-byte-stride buffers (exactly what the GEMM does) */
    unsigned short Ap[16*32], Bp[16*32];
    memset(Ap, 0, sizeof(Ap));
    memset(Bp, 0, sizeof(Bp));
    for (int r = 0; r < 16; r++) {
      for (int c = 0; c < 32; c++) {
        Ap[r*32+c] = A[r*lda + c];
        Bp[r*32+c] = B[r*ldb + c];
      }
    }

    _tile_zero(2);
    _tile_stream_loadd(0, Ap, 64);
    _tile_stream_loadd(1, Bp, 64);
    _tile_dpbf16ps(2, 0, 1);

    float C_amx[16][16];
    _tile_stored(2, C_amx, 64);

    float err = 0;
    for (int i = 0; i < 16; i++)
      for (int j = 0; j < 16; j++)
        err = fmaxf(err, fabsf(C_amx[i][j] - C_ref[i][j]));
    printf("  Max error: %.6f\n", err);
    printf("  Ref[0][0..2]: %.1f %.1f %.1f\n", C_ref[0][0], C_ref[0][1], C_ref[0][2]);
    printf("  AMX[0][0..2]: %.1f %.1f %.1f\n", C_amx[0][0], C_amx[0][1], C_amx[0][2]);
  }

  /* Test 3: same as test 2 but with contiguous layout */
  printf("\n=== Test 3: Failing data, contiguous layout ===\n");
  {
    unsigned short A[32 * 32], B[32 * 32];
    for (int m = 0; m < 32; m++)
      for (int k = 0; k < 32; k++)
        A[m*32+k] = f32_to_bf16((float)((m * 7 + k * 3) % 11 - 5));
    for (int n = 0; n < 32; n++)
      for (int k = 0; k < 32; k++)
        B[n*32+k] = f32_to_bf16((float)((n * 5 + k * 2) % 7 + 1));

    float C_ref[16][16];
    for (int i = 0; i < 16; i++)
      for (int j = 0; j < 16; j++) {
        float s = 0;
        for (int k = 0; k < 32; k++)
          s += bf16_to_f32(A[i*32+k]) * bf16_to_f32(B[j*32+k]);
        C_ref[i][j] = s;
      }

    _tile_zero(2);
    _tile_stream_loadd(0, A, 64);
    _tile_stream_loadd(1, B, 64);
    _tile_dpbf16ps(2, 0, 1);

    float C_amx[16][16];
    _tile_stored(2, C_amx, 64);

    float err = 0;
    for (int i = 0; i < 16; i++)
      for (int j = 0; j < 16; j++)
        err = fmaxf(err, fabsf(C_amx[i][j] - C_ref[i][j]));
    printf("  Max error: %.6f\n", err);
    printf("  Ref[0][0..2]: %.1f %.1f %.1f\n", C_ref[0][0], C_ref[0][1], C_ref[0][2]);
    printf("  AMX[0][0..2]: %.1f %.1f %.1f\n", C_amx[0][0], C_amx[0][1], C_amx[0][2]);
  }

  printf("\nDone.\n");
  return 0;
}
