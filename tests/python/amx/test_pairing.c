/* Test: is TDPBF16PS pairing bf16 elements incorrectly (shifted by 1)? */
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

  unsigned short bf1 = f2b(1.0f);
  unsigned short bf0 = f2b(0.0f);

  /* Pairing test: A = all 1.0, B[0] = [1, 0, 1, 0, 1, 0, ...]
     Corrent pairing (0,1)(2,3): 1*1 + 1*0 = 1 per pair, 16 pairs = 16
     Shifted by 1 (1,2)(3,4): 1*0 + 1*1 = 1 per pair, 16 pairs = 16 (same!)
     Not diagnostic. */

  /* Better test: A[0] = [0,1,1,0,0,1,1,0,...], B = all 1.0
     Corect: (0*1+1*1)=1, (1*1+0*1)=1, (0*1+1*1)=1, (1*1+0*1)=1 → 1*16 = 16
     Shift: (1*1+1*1)=2, (0*1+0*1)=0, (1*1+1*1)=2, (0*1+0*1)=0 → 2*8 + 0*8 = 16 (same!)
     Also not diagnostic with this pattern. */

  /* Pairing test: A[0] = all 1.0, B[0] = [2, 0, 2, 0, ...]
     Correct: (1*2+1*0)=2 * 16 = 32
     Shifted: (1*0+1*2)=2 * 16 = 32 (same!)
   */

  /* PAIRING TEST: A[0] alternate [1, 2, 1, 2, ...]
     Correct (0,1): A[0]*B[0]+A[1]*B[1] = 1*B[0]+2*B[1]
     Shifted (1,2): A[1]*B[1]+A[2]*B[2] = 2*B[1]+1*B[2]
     If B[1] != B[2], these differ.
     Set B[0] pattern where B[2n]=2, B[2n+1]=1:
     Correct: 1*2 + 2*1 = 4 per pair, 16*4 = 64
     Shifted: 2*1 + 1*2 = 4 → same! */

  /* UNIQUE PAIRING TEST:
     A = all 1.0. B[0] = [1, 0, 2, 0, 1, 0, 2, 0, ...]
     Correct (0,1): 1*1+1*0=1, (2,3): 1*2+1*0=2, (4,5): 1+0=1, (6,7): 2+0=2 → 8*1+8*2=24
     Shifted (1,2): 1*0+1*2=2, (3,4): 1*0+1*1=1, (5,6): 1*0+1*2=2, (7,8): 1*0+1*1=1 → 8*2+8*1=24
     Same! Argh.
   */

  /* DIAGNOSTIC PAIRING TEST:
     Use A = all 1.0 for simplicity. Choose B[0] such that shifted vs unshifted differ.
     Shift means: pair (1,2) instead of (0,1), pair (3,4) instead of (2,3), etc.
     So elements at positions 0 and 32 (non-existent) are "orphaned" at the edges.

     Correct: Σ[0..31] B[k]
     Shifted: Σ[1..32] B[k] where B[32]=0 (or B[0] from first element of next row)

     For them to differ: B[0] must differ from B[32] (0).
     So B[0] != 0 and B[32] = 0 (non-existent).

     If B[0] = 2 and all other B[k] = 1:
     Correct: 2 + 31*1 = 33
     Shifted: 31*1 + 0 = 31
     These are DIFFERENT! And my test showed PASS for this case (C[0][0]=33).
     So the shift hypothesis is WRONG. */

  /* Let me test with B[0]=[2,0,1,1,1,...] */
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;
    B[0][0] = f2b(2.0f);
    B[0][1] = bf0;

    _tile_zero(2); _tile_stream_loadd(0, A, 64); _tile_stream_loadd(1, B, 64);
    _tile_dpbf16ps(2, 0, 1);
    float C[16][16]; _tile_stored(2, C, 64);

    /* Correct: dot(A[0], B[0]) = 2+0+1*30 = 32 */
    /* Shifted: skip B[0]=2, include extra 0 at end → dot = 1+1+0+1*29 = 31 */
    float correct = 2.0f + 0.0f + 30.0f;
    float shifted = 0.0f + 1.0f + 29.0f; /* skip B[0], include B[32]=0 */
    /* Hmm, this is confusing. Let me be more precise.
       Correct: pairs (0,1)→(2,0)=2, (2,3)→(1,1)=2, ... → 2+2*15=32
       Shifted by 1: pairs (1,2)→(0,1)=1, (3,4)→(1,1)=2, (5,6)→(1,1)=2, ...
         (1,2): 0+1=1, (3,4): 1+1=2, ... 1+2*15=31
    */
    printf("B=[2,0,1,1,...]: C[0][0]=%.1f (correct=%.1f, shifted=%.1f) → %s\n",
           C[0][0], correct, shifted,
           fabs(C[0][0]-correct) < 0.1 ? "CORRECT pairing" :
           fabs(C[0][0]-shifted) < 0.1 ? "SHIFTED pairing" : "???");

    /* Check B[0] with 50% non-1.0 pattern */
    printf("  C[0][1]=%.1f (should be 32.0)\n", C[0][1]);
  }

  /* Critical test: what EXACTLY about the packed test B triggers the bug?
     I'll take the packed test B and make it work by modifying elements.
     Packed test B[0] = [1,3,5,7,2,4,6,1,3,5,7,2,4,6,1,3,5,7,2,4,6,1,3,5,7,2,4,6,1,3,5,7]

     What if I change ALL B values to the same value as B[0] (i.e., B[0]=1 everywhere)?
     Then B is all 1.0 → should PASS.

     What if I set B[0]=1, B[1]=3, rest=1 → should PASS (only 1 non-1 per row in first 2 rows).

     What if I set B values that have increasing numbers of non-1.0 elements?
   */

  /* Progressive test: increase the number of non-1.0 elements in B[0].
     Initialize B all 1.0. Then set B[0][c] = packed test pattern, one element at a time.
   */
  {
    unsigned short B_pattern[32];
    for (int c = 0; c < 32; c++)
      B_pattern[c] = f2b((float)((2*c) % 7 + 1));

    for (int n_non1 = 1; n_non1 <= 32; n_non1++) {
      cfg_tiles(16, 64);
      unsigned short A[16][32], B[16][32];
      for (int r = 0; r < 16; r++)
        for (int c = 0; c < 32; c++)
          A[r][c] = B[r][c] = bf1;

      /* Set first n_non1 elements of B[0] to the packed test pattern */
      for (int c = 0; c < n_non1 && c < 32; c++)
        B[0][c] = B_pattern[c];

      _tile_zero(2); _tile_stream_loadd(0, A, 64); _tile_stream_loadd(1, B, 64);
      _tile_dpbf16ps(2, 0, 1);
      float C[16][16]; _tile_stored(2, C, 64);

      /* Expected C[0][0] = 1*B[0][0] + 1*B[0][1] + ... + 1*1 for rest = sum of B[0] */
      float expected = 0;
      for (int c = 0; c < 32; c++)
        expected += b2f(B[0][c]);
      /* since A[0][c] = bf1 = 1.0 for all c */

      float actual = C[0][0];
      float err = fabsf(actual - expected);
      if (err > 0.1f || n_non1 <= 5 || n_non1 >= 27) {
        printf("  n_non1=%2d: C[0][0]=%8.1f expected=%8.1f %s\n",
               n_non1, actual, expected, err < 0.1 ? "OK" : "FAIL ***");
        if (err > 0.1f && n_non1 > 5) {
          /* Need more granularity */
        }
      }
    }
  }

  return 0;
}
