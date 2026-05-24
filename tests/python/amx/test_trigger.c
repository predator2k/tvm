/*
 * Find the exact bf16 value that triggers the TDPBF16PS bug.
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

static int test_tile_with_value(float a_val, int a_row, float b_val, int b_row) {
  /* Fill tile with 1.0, except one specific row has a_val/b_val */
  unsigned short A[16][32], B[16][32];
  unsigned short bf1 = f2b(1.0f), bf_a = f2b(a_val), bf_b = f2b(b_val);

  for (int r = 0; r < 16; r++)
    for (int c = 0; c < 32; c++)
      A[r][c] = B[r][c] = bf1;

  if (a_row >= 0 && a_row < 16)
    for (int c = 0; c < 32; c++)
      A[a_row][c] = bf_a;

  if (b_row >= 0 && b_row < 16)
    for (int c = 0; c < 32; c++)
      B[b_row][c] = bf_b;

  /* Reference: C[i][j] = 32.0 unless one of i,j uses special row */
  float C_ref[16][16];
  for (int i = 0; i < 16; i++) {
    float ai_row_sum = 32.0f;
    if (i == a_row) ai_row_sum = 32.0f * a_val;
    for (int j = 0; j < 16; j++) {
      float bj_row_sum = 32.0f;
      if (j == b_row) bj_row_sum = 32.0f * b_val;
      if (a_row >= 0 && b_row >= 0 && i == a_row && j == b_row)
        C_ref[i][j] = 32.0f * a_val * b_val;
      else if (a_row >= 0 && i == a_row)
        C_ref[i][j] = 32.0f * a_val * 1.0f;
      else if (b_row >= 0 && j == b_row)
        C_ref[i][j] = 32.0f * 1.0f * b_val;
      else
        C_ref[i][j] = 32.0f;
    }
  }

  _tile_zero(2);
  _tile_stream_loadd(0, A, 64);
  _tile_stream_loadd(1, B, 64);
  _tile_dpbf16ps(2, 0, 1);

  float C_amx[16][16];
  _tile_stored(2, C_amx, 64);

  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++)
      if (fabsf(C_amx[i][j] - C_ref[i][j]) > 0.1f)
        return 1;  /* Error found */
  return 0;  /* OK */
}

int main() {
  init_amx();

  /* Test 1: all 1.0 → should pass */
  cfg_tiles(16, 64);
  printf("all 1.0 (no special row):                      %s\n",
         test_tile_with_value(0, -1, 0, -1) ? "FAIL" : "PASS");

  /* Test 2-16: a_val in specific A rows */
  float test_vals[] = {3.0f, -3.0f, 4.0f, -4.0f, 5.0f, -5.0f, 6.0f, -6.0f,
                       7.0f, -7.0f, 2.0f, -2.0f, 1.0f, -1.0f, 0.5f, -0.5f};
  int n_test = sizeof(test_vals) / sizeof(test_vals[0]);

  for (int vi = 0; vi < n_test; vi++) {
    cfg_tiles(16, 64);
    int r = test_tile_with_value(test_vals[vi], 0, 1.0f, -1);
    printf("A[0]=%.1f, B=all-1.0:                         %s\n", test_vals[vi],
           r ? "FAIL ***" : "PASS");
  }

  /* Test: specific value in B row, A all 1.0 */
  for (int vi = 0; vi < n_test; vi++) {
    cfg_tiles(16, 64);
    int r = test_tile_with_value(1.0f, -1, test_vals[vi], 0);
    printf("A=all-1.0, B[0]=%.1f:                         %s\n", test_vals[vi],
           r ? "FAIL ***" : "PASS");
  }

  /* Test: the actual failing pattern (A row 0 = (3k%11-5)) */
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];
    unsigned short bf1 = f2b(1.0f);
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;

    for (int k = 0; k < 32; k++)
      A[0][k] = f2b((float)((3*k) % 11 - 5));

    printf("A[0]=pattern (3k%%11-5), B=all-1.0:           ");
    _tile_zero(2);
    _tile_stream_loadd(0, A, 64);
    _tile_stream_loadd(1, B, 64);
    _tile_dpbf16ps(2, 0, 1);
    float C_amx[16][16];
    _tile_stored(2, C_amx, 64);

    /* Reference: C[0][j] = sum(A[0][k]) = sum of pattern */
    float sum_a0 = 0;
    for (int k = 0; k < 32; k++) {
      unsigned int bits = ((unsigned int)A[0][k]) << 16;
      float f;
      memcpy(&f, &bits, sizeof(f));
      sum_a0 += f;
    }
    float ref_c00 = sum_a0 * 1.0f;
    float ref_c01 = 32.0f;
    printf("C[0][0]=%.1f (ref=%.1f), C[0][1]=%.1f (ref=%.1f)\n",
           C_amx[0][0], ref_c00, C_amx[0][1], ref_c01);
  }

  /* Test: individual A values from the pattern */
  int pattern_vals[] = {-5, -2, 1, 4, -4, -1, 2, 5, -3, 0, 3};
  for (int pvi = 0; pvi < 11; pvi++) {
    cfg_tiles(16, 64);
    float v = (float)pattern_vals[pvi];
    int r = test_tile_with_value(v, 0, 1.0f, -1);
    printf("A[0] all = %d.0:                              %s\n",
           pattern_vals[pvi], r ? "FAIL ***" : "PASS");
  }

  return 0;
}
