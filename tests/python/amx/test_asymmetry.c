/*
 * Verify asymmetry: is the bug only in src2 (B tile)?
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

static int test_pattern(const unsigned short A[16][32],
                         const unsigned short B[16][32],
                         float ref[16][16]) {
  _tile_zero(2);
  _tile_stream_loadd(0, A, 64);
  _tile_stream_loadd(1, B, 64);
  _tile_dpbf16ps(2, 0, 1);
  float C[16][16];
  _tile_stored(2, C, 64);
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++)
      if (fabsf(C[i][j] - ref[i][j]) > 0.1f)
        return 1;
  return 0;
}

int main() {
  init_amx();

  unsigned short bf1 = f2b(1.0f), bf3 = f2b(3.0f);

  /* Test 1: Exactly reproduce the small test case from test_amx_gemm */
  printf("Test 1: Reproducing 'small test' (N=16, contiguous, lda=K)...\n");
  {
    cfg_tiles(16, 64);
    int M = 32, N = 16, K = 32;
    unsigned short A[32][32];  /* M x K, contiguous */
    unsigned short B[16][32];  /* N x K, contiguous */

    for (int i = 0; i < M * K; i++)
      ((unsigned short*)A)[i] = f2b((float)(i % 7 - 3));
    for (int i = 0; i < N * K; i++)
      ((unsigned short*)B)[i] = f2b((float)(i % 5 + 1));

    /* Test first block (m=0, n=0) */
    unsigned short At[16][32], Bt[16][32];
    for (int r = 0; r < 16; r++) {
      memcpy(At[r], &A[r][0], 64);
      memcpy(Bt[r], &B[r][0], 64);
    }

    float ref[16][16];
    for (int i = 0; i < 16; i++)
      for (int j = 0; j < 16; j++) {
        float s = 0;
        for (int k = 0; k < 32; k++) {
          unsigned int ai, bj;
          memcpy(&ai, &At[i][k], 2); ai <<= 16;
          memcpy(&bj, &Bt[j][k], 2); bj <<= 16;
          float fa, fb;
          memcpy(&fa, &ai, 4);
          memcpy(&fb, &bj, 4);
          s += fa * fb;
        }
        ref[i][j] = s;
      }

    int r = test_pattern(At, Bt, ref);
    printf("  First block: %s\n", r ? "FAIL ***" : "PASS");
    printf("  A[0][0..4]: %.1f %.1f %.1f %.1f %.1f\n",
           ((float*)((unsigned int[]){((unsigned int)At[0][0])<<16}))[0],
           ((float*)((unsigned int[]){((unsigned int)At[0][1])<<16}))[0],
           ((float*)((unsigned int[]){((unsigned int)At[0][2])<<16}))[0],
           ((float*)((unsigned int[]){((unsigned int)At[0][3])<<16}))[0],
           ((float*)((unsigned int[]){((unsigned int)At[0][4])<<16}))[0]);
    printf("  B[0][0..4]: %.1f %.1f %.1f %.1f %.1f\n",
           ((float*)((unsigned int[]){((unsigned int)Bt[0][0])<<16}))[0],
           ((float*)((unsigned int[]){((unsigned int)Bt[0][1])<<16}))[0],
           ((float*)((unsigned int[]){((unsigned int)Bt[0][2])<<16}))[0],
           ((float*)((unsigned int[]){((unsigned int)Bt[0][3])<<16}))[0],
           ((float*)((unsigned int[]){((unsigned int)Bt[0][4])<<16}))[0]);
  }

  /* Test 2: A = 1.0 everywhere, B has varying values in first row only */
  printf("\nTest 2: A all 1.0, B row 0 = [3,1,3,1,...] (heterogeneous)...\n");
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;

    /* B[0] = alternating 3.0, 1.0 */
    for (int c = 0; c < 32; c++)
      B[0][c] = (c % 2 == 0) ? bf3 : bf1;

    float ref[16][16];
    for (int i = 0; i < 16; i++)
      for (int j = 0; j < 16; j++) {
        /* A[i] is all 1.0 */
        /* B[j] is all 1.0 unless j==0, then alternating 3/1 */
        if (j == 0) {
          /* dot(1_vec, [3,1,3,1,...]) = 16*3 + 16*1 = 64 */
          ref[i][j] = 64.0f;
        } else {
          ref[i][j] = 32.0f;
        }
      }

    int r = test_pattern(A, B, ref);
    printf("  %s\n", r ? "FAIL ***" : "PASS");
  }

  /* Test 3: B[0] = all 3.0, but A also has non-1.0 in row 0 */
  printf("\nTest 3: A[0]=all-5.0, B[0]=all-3.0, rest=1.0...\n");
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];
    unsigned short bf5 = f2b(-5.0f);
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;
    for (int c = 0; c < 32; c++) {
      A[0][c] = bf5;  /* A[0] = all -5.0 */
      B[0][c] = bf3;  /* B[0] = all 3.0 */
    }

    float ref[16][16];
    for (int i = 0; i < 16; i++) {
      float ai = (i == 0) ? -5.0f : 1.0f;
      for (int j = 0; j < 16; j++) {
        float bj = (j == 0) ? 3.0f : 1.0f;
        ref[i][j] = 32.0f * ai * bj;
      }
    }

    int r = test_pattern(A, B, ref);
    printf("  %s\n", r ? "FAIL ***" : "PASS");
    if (r) {
      /* Dump some results */
      _tile_zero(2);
      _tile_stream_loadd(0, A, 64);
      _tile_stream_loadd(1, B, 64);
      _tile_dpbf16ps(2, 0, 1);
      float C[16][16];
      _tile_stored(2, C, 64);
      printf("  C[0][0]=%.1f ref=%.1f, C[0][1]=%.1f ref=%.1f, C[1][0]=%.1f ref=%.1f\n",
             C[0][0], ref[0][0], C[0][1], ref[0][1], C[1][0], ref[1][0]);
    }
  }

  /* Test 4: SWAP the tiles — put problematic data in tile 0 (A) and 1.0 in tile 1 (B) */
  printf("\nTest 4: SWAP: load failing data into A (src1, tmm0), keep B all 1.0...\n");
  {
    cfg_tiles(16, 64);
    unsigned short A[16][32], B[16][32];
    for (int r = 0; r < 16; r++)
      for (int c = 0; c < 32; c++)
        A[r][c] = B[r][c] = bf1;

    /* A[0] = the original failing B pattern: (n*5 + k*2) % 7 + 1 */
    for (int c = 0; c < 32; c++)
      A[0][c] = f2b((float)((0*5 + c*2) % 7 + 1));

    float ref[16][16];
    for (int i = 0; i < 16; i++)
      for (int j = 0; j < 16; j++) {
        float ai_val = (i == 0) ? ((2*j) % 7 + 1) : 1.0f; // WRONG - need to compute properly
        ref[i][j] = 32.0f;
        if (i == 0) {
          float s = 0;
          for (int k = 0; k < 32; k++)
            s += ((float)(int)((2*k) % 7 + 1)) * 1.0f;
          ref[i][j] = s;
        }
      }

    int r = test_pattern(A, B, ref);
    printf("  %s\n", r ? "FAIL ***" : "PASS");
    if (r) printf("  Bug also affects src1!\n");
  }

  return 0;
}
