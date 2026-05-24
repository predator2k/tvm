/* Test the "pair" hypothesis: can only handle 1 non-standard pair per B row? */
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

  /* Test 1: one non-standard PAIR (0,1) = (2,3) */
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;
    B[0][0] = bf2;
    B[0][1] = bf3;

    _tile_zero(2); _tile_stream_loadd(0, A, 64); _tile_stream_loadd(1, B, 64);
    _tile_dpbf16ps(2, 0, 1);
    float C[16][16]; _tile_stored(2, C, 64);

    /* B[0] = [2,3,1,1,...,1]. Sum = 2+3+30 = 35. C[0][0] should be 35 (with A all 1.0). */
    printf("B pair(0,1)=(2,3) only: C[0][0]=%.1f (exp=35.0) %s\n",
           C[0][0], fabsf(C[0][0]-35.0)<0.1 ? "OK" : "FAIL ***");
  }

  /* Test 2: non-standard pair at (6,7), rest=1 */
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;
    B[0][6] = bf2;
    B[0][7] = bf3;

    _tile_zero(2); _tile_stream_loadd(0, A, 64); _tile_stream_loadd(1, B, 64);
    _tile_dpbf16ps(2, 0, 1);
    float C[16][16]; _tile_stored(2, C, 64);

    /* Sum = 2+3+30 = 35 */
    printf("B pair(6,7)=(2,3) only: C[0][0]=%.1f (exp=35.0) %s\n",
           C[0][0], fabsf(C[0][0]-35.0)<0.1 ? "OK" : "FAIL ***");
  }

  /* Test 3: two non-standard pairs: (0,1)=(2,3) and (6,7)=(2,3) */
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;
    B[0][0] = bf2; B[0][1] = bf3;
    B[0][6] = bf2; B[0][7] = bf3;

    _tile_zero(2); _tile_stream_loadd(0, A, 64); _tile_stream_loadd(1, B, 64);
    _tile_dpbf16ps(2, 0, 1);
    float C[16][16]; _tile_stored(2, C, 64);

    /* Sum = 2+3+2+3+28 = 38 */
    printf("B TWO pairs (0,1) and (6,7) = (2,3): C[0][0]=%.1f (exp=38.0) %s\n",
           C[0][0], fabsf(C[0][0]-38.0)<0.1 ? "OK" : "FAIL ***");
  }

  /* Test 4: non-standard at (1,2) — spanning pairs! */
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;
    B[0][1] = bf2;
    B[0][2] = bf3;

    _tile_zero(2); _tile_stream_loadd(0, A, 64); _tile_stream_loadd(1, B, 64);
    _tile_dpbf16ps(2, 0, 1);
    float C[16][16]; _tile_stored(2, C, 64);

    /* B[0] = [1,2,3,1,1,...,1]. Sum = 1+2+3+29 = 35.
       But pairs: (0,1)=(1,2), (2,3)=(3,1) → TWO pairs have non-1.0 elements! */
    printf("B at (1,2)=(2,3) spanning pairs: C[0][0]=%.1f (exp=35.0) %s\n",
           C[0][0], fabsf(C[0][0]-35.0)<0.1 ? "OK" : "FAIL ***");
  }

  /* Test 5: just one non-1.0 in the SECOND position of a pair */
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;
    B[0][1] = bf2;  /* second element of pair (0,1) is 2 */

    _tile_zero(2); _tile_stream_loadd(0, A, 64); _tile_stream_loadd(1, B, 64);
    _tile_dpbf16ps(2, 0, 1);
    float C[16][16]; _tile_stored(2, C, 64);

    /* B[0] = [1,2,1,...,1]. Sum = 1+2+30 = 33 */
    printf("B[0][1]=2.0 only: C[0][0]=%.1f (exp=33.0) %s\n",
           C[0][0], fabsf(C[0][0]-33.0)<0.1 ? "OK" : "FAIL ***");
  }

  /* Test 6: non-1.0 values in TWO different rows of B */
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;
    B[0][1] = bf2;
    B[1][1] = bf3;

    _tile_zero(2); _tile_stream_loadd(0, A, 64); _tile_stream_loadd(1, B, 64);
    _tile_dpbf16ps(2, 0, 1);
    float C[16][16]; _tile_stored(2, C, 64);

    /* C[0][1] = dot(A[0], B[1]) = 1+3+30 = 34 */
    printf("B[0][1]=2, B[1][1]=3 (two rows, one each): C[0][1]=%.1f (exp=34.0) %s\n",
           C[0][1], fabsf(C[0][1]-34.0)<0.1 ? "OK" : "FAIL ***");
    printf("   also C[0][0] = dot(A[0],B[0]) = %.1f (exp=33.0) %s\n",
           C[0][0], fabsf(C[0][0]-33.0)<0.1 ? "OK" : "FAIL ***");
  }

  return 0;
}
