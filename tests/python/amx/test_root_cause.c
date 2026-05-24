/* Find the exact trigger: check load-store for B tile, and which B elements cause failure. */
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
  cfg_tiles(16, 64);

  unsigned short bf1 = f2b(1.0f);

  /* Test: B tile with progressively more non-1.0 elements */
  /* Check 1: B[0][0]=2.0, rest of B=1.0 */
  {
    unsigned short A[16][32], B[16][32];
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;
    B[0][0] = f2b(2.0f);

    /* Verify B load roundtrip */
    _tile_zero(0);
    _tile_stream_loadd(0, A, 64);
    _tile_zero(1);
    _tile_stream_loadd(1, B, 64);
    unsigned short Bcheck[16][32];
    _tile_stored(1, Bcheck, 64);
    int load_ok = (Bcheck[0][0] == B[0][0]) && (Bcheck[0][1] == B[0][1]);
    printf("B[0][0]=2.0, rest=1.0: load roundtrip %s\n", load_ok ? "OK" : "FAIL");

    /* Run TDPBF16PS */
    _tile_zero(2);
    _tile_dpbf16ps(2, 0, 1);
    float C[16][16];
    _tile_stored(2, C, 64);

    /* Expected: C[i][j] = dot(A[i], B[j])
       A[i] = all 1.0 (32 elements) → row sum = 32
       B[0] = [2, 1, 1, ..., 1] → row sum = 33
       B[j>0] = all 1.0 → row sum = 32
       C[i][0] = dot(all-1, B[0]) = 33.0
       C[i][j>0] = dot(all-1, B[j]) = 32.0
    */
    float expected_C00 = 33.0f;
    float expected_C01 = 32.0f;
    printf("  C[0][0]=%.1f (exp=%.1f) %s, C[0][1]=%.1f (exp=%.1f) %s\n",
           C[0][0], expected_C00, fabsf(C[0][0]-expected_C00)<0.1?"OK":"FAIL***",
           C[0][1], expected_C01, fabsf(C[0][1]-expected_C01)<0.1?"OK":"FAIL***");
  }

  /* Check 2: B[0][0]=1.0, B[0][1]=2.0, rest=1.0 → first element IS 1.0 */
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;
    B[0][0] = bf1;      /* first element stays 1.0 */
    B[0][1] = f2b(2.0f); /* second element = 2.0 */

    /* Verify B load */
    _tile_zero(0); _tile_stream_loadd(0, A, 64);
    _tile_zero(1); _tile_stream_loadd(1, B, 64);
    unsigned short Bcheck[16][32];
    _tile_stored(1, Bcheck, 64);
    int load_ok = (Bcheck[0][0] == B[0][0]) && (Bcheck[0][1] == B[0][1]);
    printf("\nB[0]=[1,2,1,1,...]: load roundtrip %s\n", load_ok ? "OK" : "FAIL");

    _tile_zero(2);
    _tile_dpbf16ps(2, 0, 1);
    float C[16][16];
    _tile_stored(2, C, 64);

    /* B[0] row sum = 1+2+1*30 = 33. Same as before. Check C[1][0] too (B[0] dot A[1]=all-1) */
    printf("  C[0][0]=%.1f (exp=33.0) %s, C[1][0]=%.1f (exp=33.0) %s, C[0][1]=%.1f (exp=32.0) %s\n",
           C[0][0], fabsf(C[0][0]-33.0f)<0.1?"OK":"FAIL***",
           C[1][0], fabsf(C[1][0]-33.0f)<0.1?"OK":"FAIL***",
           C[0][1], fabsf(C[0][1]-32.0f)<0.1?"OK":"FAIL***");
  }

  /* Check 3: exactly reproduce the small test's B and A patterns */
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];

    /* Small test data (flat fill): A[i] = bf16(i%7-3), B[i] = bf16(i%5+1) */
    for (int r = 0; r < 16; r++) {
      for (int c = 0; c < 32; c++) {
        int ai = r * 32 + c;
        int bi = r * 32 + c;
        A[r][c] = f2b((float)(ai % 7 - 3));
        B[r][c] = f2b((float)(bi % 5 + 1));
      }
    }

    printf("\nSmall test pattern (A=%%7-3, B=%%5+1):\n");
    printf("  B[0][0..5]: %.1f %.1f %.1f %.1f %.1f %.1f\n",
           b2f(B[0][0]), b2f(B[0][1]), b2f(B[0][2]),
           b2f(B[0][3]), b2f(B[0][4]), b2f(B[0][5]));

    /* Verify B load */
    _tile_zero(0); _tile_stream_loadd(0, A, 64);
    _tile_zero(1); _tile_stream_loadd(1, B, 64);
    unsigned short Bcheck[16][32];
    _tile_stored(1, Bcheck, 64);
    int load_ok = 1;
    for (int c = 0; c < 5; c++)
      if (Bcheck[0][c] != B[0][c]) load_ok = 0;
    printf("  B load roundtrip: %s\n", load_ok ? "OK" : "FAIL");

    _tile_zero(2);
    _tile_dpbf16ps(2, 0, 1);
    float C[16][16];
    _tile_stored(2, C, 64);

    /* Reference first few */
    float C00_ref = 0, C01_ref = 0, C10_ref = 0;
    for (int k = 0; k < 32; k++) {
      C00_ref += b2f(A[0][k]) * b2f(B[0][k]);
      C01_ref += b2f(A[0][k]) * b2f(B[1][k]);
      C10_ref += b2f(A[1][k]) * b2f(B[0][k]);
    }
    printf("  C[0][0]=%.1f (ref=%.1f) %s\n", C[0][0], C00_ref,
           fabsf(C[0][0]-C00_ref)<0.1?"OK":"FAIL***");
    printf("  C[0][1]=%.1f (ref=%.1f) %s\n", C[0][1], C01_ref,
           fabsf(C[0][1]-C01_ref)<0.1?"OK":"FAIL***");
    printf("  C[1][0]=%.1f (ref=%.1f) %s\n", C[1][0], C10_ref,
           fabsf(C[1][0]-C10_ref)<0.1?"OK":"FAIL***");
  }

  /* Check 4: packed test pattern */
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];

    for (int r = 0; r < 16; r++) {
      for (int c = 0; c < 32; c++) {
        A[r][c] = f2b((float)((r * 7 + c * 3) % 11 - 5));
        B[r][c] = f2b((float)((r * 5 + c * 2) % 7 + 1));
      }
    }

    printf("\nPacked test pattern (A=(m*7+k*3)%%11-5, B=(n*5+k*2)%%7+1):\n");
    printf("  B[0][0..5]: %.1f %.1f %.1f %.1f %.1f %.1f\n",
           b2f(B[0][0]), b2f(B[0][1]), b2f(B[0][2]),
           b2f(B[0][3]), b2f(B[0][4]), b2f(B[0][5]));
    printf("  A[0][0..5]: %.1f %.1f %.1f %.1f %.1f %.1f\n",
           b2f(A[0][0]), b2f(A[0][1]), b2f(A[0][2]),
           b2f(A[0][3]), b2f(A[0][4]), b2f(A[0][5]));

    /* Verify B load */
    _tile_zero(0); _tile_stream_loadd(0, A, 64);
    _tile_zero(1); _tile_stream_loadd(1, B, 64);
    unsigned short Bcheck[16][32];
    _tile_stored(1, Bcheck, 64);
    int load_ok = 1;
    for (int c = 0; c < 5; c++)
      if (Bcheck[0][c] != B[0][c]) load_ok = 0;
    printf("  B load roundtrip: %s\n", load_ok ? "OK" : "FAIL");

    _tile_zero(2);
    _tile_dpbf16ps(2, 0, 1);
    float C[16][16];
    _tile_stored(2, C, 64);

    float C00_ref = 0, C01_ref = 0, C10_ref = 0;
    for (int k = 0; k < 32; k++) {
      C00_ref += b2f(A[0][k]) * b2f(B[0][k]);
      C01_ref += b2f(A[0][k]) * b2f(B[1][k]);
      C10_ref += b2f(A[1][k]) * b2f(B[0][k]);
    }
    printf("  C[0][0]=%.1f (ref=%.1f) %s\n", C[0][0], C00_ref,
           fabsf(C[0][0]-C00_ref)<0.1?"OK":"FAIL***");
    printf("  C[0][1]=%.1f (ref=%.1f) %s\n", C[0][1], C01_ref,
           fabsf(C[0][1]-C01_ref)<0.1?"OK":"FAIL***");
    printf("  C[1][0]=%.1f (ref=%.1f) %s\n", C[1][0], C10_ref,
           fabsf(C[1][0]-C10_ref)<0.1?"OK":"FAIL***");
  }

  return 0;
}
