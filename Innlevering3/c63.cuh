#ifndef C63
#define C63

#ifdef WIN32
#include<cstdint>
#include<Windows.h>
#else
#include <inttypes.h>
#endif
#include <stdint.h>
#include <stdio.h>

#define MAX_FILELENGTH 200
#define DEFAULT_OUTPUT_FILE "a.mjpg"

#define COLOR_COMPONENTS 3

#define YX 2
#define YY 2
#define UX 1
#define UY 1
#define VX 1
#define VY 1

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

struct yuv
{
  uint8_t *Y;
  uint8_t *U;
  uint8_t *V;
};

struct dct
{
  int16_t *Ydct;
  int16_t *Udct;
  int16_t *Vdct;
};

typedef struct yuv yuv_t;
typedef struct dct dct_t;

struct entropy_ctx
{
  FILE *fp;
  unsigned int bit_buffer;
  unsigned int bit_buffer_width;
};

struct macroblock
{
  int use_mv;
  int8_t mv_x, mv_y;
};

//Variables for domain partitioning
static int split_atY;
static int gpu_partY;
static int cpu_partY;
static int split_atUV;
static int gpu_partUV;
static int cpu_partUV;

/* Definitions are found in 'io.c' */
int read_bytes(FILE *fp, void *data, unsigned int sz);
uint16_t get_bits(struct entropy_ctx *c, uint8_t n);
uint8_t get_byte(FILE *fp);
void flush_bits(struct entropy_ctx *c);
void put_bits(struct entropy_ctx *c, uint16_t bits, uint8_t n);
void put_byte(FILE *fp, int byte);
void put_bytes(FILE *fp, const void* data, unsigned int len);

/* Definitions are found in 'dsp.c' */
void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl);
void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl);
void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result);

/* Definitions are found in 'common.c' */
void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, int16_t *out_data, uint8_t *quantization);
void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, uint8_t *out_data, uint8_t *quantization);
void destroy_frame(struct frame *f);
void dump_image(yuv_t *image, int w, int h, FILE *fp);

/* Definitions are found in 'me.c' */
void c63_motion_estimate(struct c63_common_cpu *cm);
void c63_motion_compensate(struct c63_common_cpu *cm);

#endif