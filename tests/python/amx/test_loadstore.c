/*
 * Round-trip test: load data into tile, store back, verify.
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
  if (!(bitmask & (1 << 18)))
    syscall(SYS_arch_prctl, 0x1023, 18);
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

/* Load 16x32 bf16 from src (stride=src_stride bytes), store to dst (stride=64 bytes) */
static void load_store_roundtrip(unsigned short *dst_contig, const unsigned short *src,
                                  int src_stride_elems) {
  /* Pack src data into a local buffer with 64-byte-per-row stride */
  unsigned short packed[16 * 32];
  for (int r = 0; r < 16; r++)
    for (int c = 0; c < 32; c++)
      packed[r * 32 + c] = src[r * src_stride_elems + c];

  _tile_zero(0);
  _tile_stream_loadd(0, packed, 64);
  _tile_stored(0, dst_contig, 64);
}

int main() {
  init_amx();
  config(16, 64);

  /* Create test data: matrix with 128 cols, only first 32 used */
  const int outer_stride = 128;
  unsigned short *src = calloc(16 * outer_stride, sizeof(unsigned short));
  for (int r = 0; r < 16; r++)
    for (int c = 0; c < 32; c++)
      src[r * outer_stride + c] = (unsigned short)((r * 32 + c) & 0xFFFF);

  /* Round trip through AMX tile */
  unsigned short *dst = calloc(16 * 32, sizeof(unsigned short));
  load_store_roundtrip(dst, src, outer_stride);

  /* Compare */
  int errors = 0;
  for (int r = 0; r < 16 && errors < 10; r++) {
    for (int c = 0; c < 32 && errors < 10; c++) {
      unsigned short expected = src[r * outer_stride + c];
      unsigned short got = dst[r * 32 + c];
      if (expected != got) {
        printf("  MISMATCH [%d][%d]: expected=%u got=%u\n", r, c, expected, got);
        errors++;
      }
    }
  }
  printf("Load-store roundtrip (outer_stride=%d): %s (%d errors)\n",
         outer_stride, errors ? "FAIL" : "PASS", errors);

  /* Also test with contiguous source (stride=32) */
  unsigned short *src2 = calloc(16 * 32, sizeof(unsigned short));
  for (int i = 0; i < 16 * 32; i++)
    src2[i] = (unsigned short)(i & 0xFFFF);

  unsigned short *dst2 = calloc(16 * 32, sizeof(unsigned short));
  load_store_roundtrip(dst2, src2, 32);

  errors = 0;
  for (int i = 0; i < 16 * 32 && errors < 10; i++) {
    if (src2[i] != dst2[i]) {
      printf("  MISMATCH contig[%d]: expected=%u got=%u\n", i, src2[i], dst2[i]);
      errors++;
    }
  }
  printf("Load-store roundtrip (contiguous): %s (%d errors)\n",
         errors ? "FAIL" : "PASS", errors);

  free(src); free(dst); free(src2); free(dst2);
  return errors > 0;
}
