/*
 * Minimal AMX sanity check: load tiles, dpbf16ps, store back.
 */
#include <errno.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

static void init_amx() {
  uint64_t bitmask = 0;
  syscall(SYS_arch_prctl, 0x1022, &bitmask);
  if (!(bitmask & (1 << 18))) {
    syscall(SYS_arch_prctl, 0x1023, 18);
  }
}

static void config(int rows, int colsb) {
  unsigned char cfg[64] = {0};
  cfg[0] = 1;
  for (int i = 0; i < 8; i++) {
    *(uint16_t *)(cfg + 16 + 2 * i) = (uint16_t)colsb;
    cfg[48 + i] = (uint8_t)rows;
  }
  _tile_loadconfig(cfg);
}

static inline unsigned short f2b(float v) {
  unsigned int bits;
  memcpy(&bits, &v, sizeof(bits));
  return (unsigned short)(bits >> 16);
}

static inline float b2f(unsigned short v) {
  unsigned int bits = ((unsigned int)v) << 16;
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

int main() {
  init_amx();
  config(16, 64);

  /* Test: C[16][16] = A[16][32] @ B[16][32]^T where all values are 1.0 */
  unsigned short A[16 * 32];
  unsigned short B[16 * 32];
  unsigned short bf16_one = f2b(1.0f);

  for (int i = 0; i < 16 * 32; i++) {
    A[i] = bf16_one;
    B[i] = bf16_one;
  }

  /* Reference: each element of C should be 32.0 (sum of 32 (1.0*1.0) pairs) */
  float expected = 32.0f;

  _tile_zero(2);
  _tile_stream_loadd(0, A, 64);
  _tile_stream_loadd(1, B, 64);
  _tile_dpbf16ps(2, 0, 1);

  float C_out[16 * 16] = {0};
  /* Stride = 16 fp32 elements * 4 bytes = 64 */
  _tile_stored(2, C_out, 64);

  int errors = 0;
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      float diff = C_out[i * 16 + j] - expected;
      if (diff < -0.001f || diff > 0.001f) {
        if (errors < 5)
          printf("  C[%d][%d] = %.6f (expected %.6f)\n", i, j, C_out[i * 16 + j], expected);
        errors++;
      }
    }
  }

  if (errors)
    printf("FAIL: %d errors out of 256 elements\n", errors);
  else
    printf("PASS: All ones test\n");

  /* Test 2: identity check — A has row i = e_i, B has row j = e_j */
  unsigned short A2[16 * 32] = {0};
  unsigned short B2[16 * 32] = {0};

  /* A2: row 0 = all ones, row 1 = all twos, etc. */
  for (int r = 0; r < 16; r++)
    for (int c = 0; c < 32; c++)
      A2[r * 32 + c] = f2b((float)(r + 1));

  /* B2: row 0 = all ones, row 1 = all ones, etc. (same for all rows) */
  for (int r = 0; r < 16; r++)
    for (int c = 0; c < 32; c++)
      B2[r * 32 + c] = bf16_one;

  float C2_expected[16][16];
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++)
      C2_expected[i][j] = 32.0f * (i + 1);

  _tile_zero(2);
  _tile_stream_loadd(0, A2, 64);
  _tile_stream_loadd(1, B2, 64);
  _tile_dpbf16ps(2, 0, 1);

  float C2_out[16 * 16] = {0};
  _tile_stored(2, C2_out, 64);

  errors = 0;
  float max_err = 0;
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      float diff = C2_out[i * 16 + j] - C2_expected[i][j];
      if (diff < -0.001f || diff > 0.001f) {
        if (errors < 5)
          printf("  C2[%d][%d] = %.6f (expected %.6f)\n", i, j,
                 C2_out[i * 16 + j], C2_expected[i][j]);
        errors++;
      }
      if (diff > max_err) max_err = diff;
      if (-diff > max_err) max_err = -diff;
    }
  }

  if (errors)
    printf("FAIL: %d errors out of 256 (max_err=%.6f)\n", errors, max_err);
  else
    printf("PASS: Scaled rows test\n");

  return errors > 0;
}
