/* Test whether _tile_loadd fixes the TDPBF16PS pair processing bug. */
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

int main() {
  init_amx();

  unsigned short bf1 = f2b(1.0f), bf2 = f2b(2.0f), bf3 = f2b(3.0f);
  unsigned short bf5 = f2b(5.0f);

  /* Test with _tile_loadd (non-streaming) instead of _tile_stream_loadd */
  /* Use the packed test pattern where the bug is known to occur */
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;

    /* B[0] has MANY non-1.0 values — the packed test pattern */
    for (int c = 0; c < 32; c++)
      B[0][c] = f2b((float)((2*c) % 7 + 1));

    _tile_zero(2);

    /* Use _tile_loadd (NOT streaming) */
    _tile_loadd(0, A, 64);
    _tile_loadd(1, B, 64);

    _tile_dpbf16ps(2, 0, 1);

    float C_loadd[16][16];
    _tile_stored(2, C_loadd, 64);

    float expected = 0;
    for (int c = 0; c < 32; c++)
      expected += b2f(B[0][c]);  /* since A[0][c] = 1.0 */

    printf("_tile_loadd:      C[0][0]=%.1f (expected=%.1f) %s\n",
           C_loadd[0][0], expected,
           fabsf(C_loadd[0][0] - expected) < 0.1 ? "OK" : "FAIL ***");
  }

  /* Now test with _tile_stream_loadd for comparison */
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;
    for (int c = 0; c < 32; c++)
      B[0][c] = f2b((float)((2*c) % 7 + 1));

    _tile_zero(2);
    _tile_stream_loadd(0, A, 64);
    _tile_stream_loadd(1, B, 64);
    _tile_dpbf16ps(2, 0, 1);

    float C_stream[16][16];
    _tile_stored(2, C_stream, 64);

    float expected = 0;
    for (int c = 0; c < 32; c++)
      expected += b2f(B[0][c]);

    printf("_tile_stream_loadd: C[0][0]=%.1f (expected=%.1f) %s\n",
           C_stream[0][0], expected,
           fabsf(C_stream[0][0] - expected) < 0.1 ? "OK" : "FAIL ***");
  }

  /* Test same thing but with A having the pattern and B all 1.0 */
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;
    for (int c = 0; c < 32; c++)
      A[0][c] = f2b((float)((2*c) % 7 + 1));  /* pattern in A, not B */

    _tile_zero(2);
    _tile_loadd(0, A, 64);
    _tile_loadd(1, B, 64);
    _tile_dpbf16ps(2, 0, 1);
    float C[16][16];
    _tile_stored(2, C, 64);

    float sum_A = 0;
    for (int c = 0; c < 32; c++)
      sum_A += b2f(A[0][c]);

    printf("A=pattern,B=all1 (_tile_loadd): C[0][0]=%.1f (expected=%.1f) %s\n",
           C[0][0], sum_A,
           fabsf(C[0][0] - sum_A) < 0.1 ? "OK" : "FAIL ***");
  }

  /* Also check: does the problem manifest as treating non-first-pair elements as 1.0?
     Test with pai (6,7)=(2,3) using _tile_loadd */
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;
    B[0][6] = bf2;
    B[0][7] = bf3;

    _tile_zero(2);
    _tile_loadd(0, A, 64);
    _tile_loadd(1, B, 64);
    _tile_dpbf16ps(2, 0, 1);
    float C[16][16];
    _tile_stored(2, C, 64);

    /* Total should be 32 + (2-1) + (3-1) = 35 */
    printf("pair(6,7)=(2,3) _tile_loadd: C[0][0]=%.1f (expected=35.0) %s\n",
           C[0][0], fabsf(C[0][0]-35.0)<0.1 ? "OK" : "FAIL ***");
  }

  return 0;
}
