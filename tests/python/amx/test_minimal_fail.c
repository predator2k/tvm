/*
 * Find minimal failing case: feed just the first row of problematic data.
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

/* Compute reference dot product between two 32-element bf16 vectors (stored as uint16) */
static float dot_ref(const unsigned short *a, const unsigned short *b, int n) {
  float s = 0;
  for (int i = 0; i < n; i++)
    s += b2f(a[i]) * b2f(b[i]);
  return s;
}

int main() {
  init_amx();
  cfg_tiles(16, 64);

  /* Fill tiles where rows 0 and 1 are the "failing" data, rest are 1.0 */
  unsigned short A[16][32], B[16][32];
  unsigned short bf16_one = f2b(1.0f);

  /* Fill ALL rows with 1.0 first */
  for (int r = 0; r < 16; r++)
    for (int c = 0; c < 32; c++)
      A[r][c] = B[r][c] = bf16_one;

  printf("Testing individual rows with failing-data row 0...\n");

  /* Fill A row 0 and B row 0 with failing data */
  for (int k = 0; k < 32; k++) {
    A[0][k] = f2b((float)((0 * 7 + k * 3) % 11 - 5));
    B[0][k] = f2b((float)((0 * 5 + k * 2) % 7 + 1));
  }

  /* Reference: each C[i][j] = dot(A_row_i, B_row_j) */
  float C_ref[16][16];
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++)
      C_ref[i][j] = dot_ref(A[i], B[j], 32);

  /* AMX */
  _tile_zero(2);
  _tile_stream_loadd(0, A, 64);
  _tile_stream_loadd(1, B, 64);
  _tile_dpbf16ps(2, 0, 1);

  float C_amx[16][16];
  _tile_stored(2, C_amx, 64);

  printf("  C[0][0] ref=%.1f amx=%.1f\n", C_ref[0][0], C_amx[0][0]);
  printf("  C[0][1] ref=%.1f amx=%.1f\n", C_ref[0][1], C_amx[0][1]);

  /* Check: which elements are wrong? */
  int errors = 0;
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++)
      if (fabsf(C_ref[i][j] - C_amx[i][j]) > 0.01f) {
        if (errors < 10)
          printf("  Wrong: C[%d][%d] ref=%.1f amx=%.1f (B_row[%d] starts: %.1f %.1f %.1f)\n",
                 i, j, C_ref[i][j], C_amx[i][j], j,
                 b2f(B[j][0]), b2f(B[j][1]), b2f(B[j][2]));
        errors++;
      }
  printf("  Total errors: %d / 256\n", errors);

  /* Test: what if we ONLY fill row 0 of A and B with failing data, but ALL others with 1.0? */
  printf("\n--- All 1.0 baseline ---\n");
  for (int r = 0; r < 16; r++)
    for (int c = 0; c < 32; c++)
      A[r][c] = B[r][c] = bf16_one;

  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++)
      C_ref[i][j] = 32.0f;  // dot of two all-1 vectors of length 32

  _tile_zero(2);
  _tile_stream_loadd(0, A, 64);
  _tile_stream_loadd(1, B, 64);
  _tile_dpbf16ps(2, 0, 1);
  _tile_stored(2, C_amx, 64);

  errors = 0;
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++)
      if (fabsf(C_amx[i][j] - 32.0f) > 0.01f) errors++;
  printf("  All-ones: %d errors\n", errors);

  /* Check specifically: what does the tile contain after the first test? */
  printf("\n--- Did tile 0 survive after dpbf16ps? ---\n");

  /* Fill A and B with the failing data */
  for (int k = 0; k < 32; k++) {
    A[0][k] = f2b((float)((3*k) % 11 - 5));
    B[0][k] = f2b((float)((2*k) % 7 + 1));
  }
  /* Keep rows 1..15 as 1.0 */
  for (int r = 1; r < 16; r++)
    for (int c = 0; c < 32; c++)
      A[r][c] = B[r][c] = bf16_one;

  /* Load A and B, then store A back to check */
  _tile_zero(0);
  _tile_stream_loadd(0, A, 64);
  _tile_zero(1);
  _tile_stream_loadd(1, B, 64);

  /* Store tile 0 back to verify load correctness */
  unsigned short A_back[16][32];
  _tile_stored(0, A_back, 64);

  printf("  A load-roundtrip check:\n");
  for (int r = 0; r < 3; r++) {
    printf("    Row %d: A[%d][0..3] = %.1f %.1f %.1f %.1f  |  A_back = %.1f %.1f %.1f %.1f\n",
           r, r, b2f(A[r][0]), b2f(A[r][1]), b2f(A[r][2]), b2f(A[r][3]),
           b2f(A_back[r][0]), b2f(A_back[r][1]), b2f(A_back[r][2]), b2f(A_back[r][3]));
  }

  /* Now run TDPBF16PS and check result */
  _tile_zero(2);
  _tile_dpbf16ps(2, 0, 1);
  _tile_stored(2, C_amx, 64);

  /* Reference */
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++)
      C_ref[i][j] = dot_ref(A[i], B[j], 32);

  errors = 0;
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++)
      if (fabsf(C_ref[i][j] - C_amx[i][j]) > 0.01f) {
        if (errors < 5)
          printf("  Wrong: C[%d][%d] ref=%.1f amx=%.1f\n", i, j, C_ref[i][j], C_amx[i][j]);
        errors++;
      }
  printf("  After roundtrip check: %d errors\n", errors);

  return errors > 0;
}
