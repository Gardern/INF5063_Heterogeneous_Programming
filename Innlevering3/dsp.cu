#ifdef WIN32
#include<cstdint>
#include<boost\math\special_functions\round.hpp>
#else
#include <inttypes.h>
#include <math.h>
#include <inttypes.h>
#include <math.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<x86intrin.h>
#include<immintrin.h>
#endif
#include <stdlib.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "tables.cuh"

#define ISQRT2 0.70710678118654f

static void transpose_block(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      out_data[i*8+j] = in_data[j*8+i];
    }
  }
}

static void dct_1d(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    float dct = 0;

    for (j = 0; j < 8; ++j)
    {
      dct += in_data[j] * dctlookup[j][i];
    }

    out_data[i] = dct;
  }
}

static void idct_1d(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    float idct = 0;

    for (j = 0; j < 8; ++j)
    {
      idct += in_data[j] * dctlookup[i][j];
    }

    out_data[i] = idct;
  }
}

static void scale_block(float *in_data, float *out_data)
{
  int u, v;

  for (v = 0; v < 8; ++v)
  {
    for (u = 0; u < 8; ++u)
    {
      float a1 = !u ? ISQRT2 : 1.0f;
      float a2 = !v ? ISQRT2 : 1.0f;

      /* Scale according to normalizing function */
      out_data[v*8+u] = in_data[v*8+u] * a1 * a2;
    }
  }
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[v*8+u];

    /* Zig-zag and quantize */
#ifdef WIN32
	out_data[zigzag] = (float) boost::math::round((dct / 4.0) / quant_tbl[zigzag]);
#else
    out_data[zigzag] = (float) round((dct / 4.0) / quant_tbl[zigzag]);
#endif
  }
}

static void dequantize_block(float *in_data, float *out_data,
    uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[zigzag];

    /* Zig-zag and de-quantize */
#ifdef WIN32
	out_data[v*8+u] = (float) boost::math::round((dct * quant_tbl[zigzag]) / 4.0);
#else
    out_data[v*8+u] = (float) round((dct * quant_tbl[zigzag]) / 4.0);
#endif
  }
}

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
#ifndef WIN32
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));
#else
	__declspec(align(16)) float mb[8*8];
	__declspec(align(16)) float mb2[8*8];
#endif

  int i, v;

  for (i = 0; i < 64; ++i) { mb2[i] = in_data[i]; }

  /* Two 1D DCT operations with transpose */
  for (v = 0; v < 8; ++v) { dct_1d(mb2+v*8, mb+v*8); }
  transpose_block(mb, mb2);
  for (v = 0; v < 8; ++v) { dct_1d(mb2+v*8, mb+v*8); }
  transpose_block(mb, mb2);

  scale_block(mb2, mb);
  quantize_block(mb, mb2, quant_tbl);

  for (i = 0; i < 64; ++i) { out_data[i] = mb2[i]; }
}

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
#ifndef WIN32
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));
#else
	__declspec(align(16)) float mb[8*8];
	__declspec(align(16)) float mb2[8*8];
#endif

  int i, v;

  for (i = 0; i < 64; ++i) { mb[i] = in_data[i]; }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);

  /* Two 1D inverse DCT operations with transpose */
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);

  for (i = 0; i < 64; ++i) { out_data[i] = mb[i]; }
}

#define BEST_SIMD

#ifdef BEST_SIMD
void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
  int v;

  *result = 0;

  __m128i a_b2, a_b1, b_b2, b_b1, tempResult;
  __m128i total;

  for (v = 0; v < 8; ++v)
  {
	//Load in 8 unsigned char from block2 and block1 into registers
	b_b2 = _mm_loadl_epi64((__m128i*)&block2[v*stride]);
	b_b1 = _mm_loadl_epi64((__m128i*)&block1[v*stride]);

	//Load in the next 8 unsigned char from block2 and block1 into registers,
	//and put them together with the previous registers
    v++;
#ifndef WIN32
    a_b2 = (__m128i) _mm_loadh_pd((__m128d)b_b2, (double*)&block2[v*stride]);
    a_b1 = (__m128i) _mm_loadh_pd((__m128d)b_b1, (double*)&block1[v*stride]);
#else
	a_b2 = _mm_castpd_si128(_mm_loadh_pd(_mm_castsi128_pd(b_b2), (double*)&block2[v*stride]));
    a_b1 = _mm_castpd_si128(_mm_loadh_pd(_mm_castsi128_pd(b_b1), (double*)&block1[v*stride]));
#endif
    //Compute the absolute difference
    tempResult = _mm_sad_epu8(a_b2, a_b1);
    total = _mm_add_epi16(total, tempResult);
  }
  *result += _mm_extract_epi16(total, 0);
  *result += _mm_extract_epi16(total, 4);
}
#else
void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
	int v, u;
	u = 0;
	__m128i a, b, c;
	short tempResult = 0;
	*result = 0;

	for(v = 0; v < 8; ++v)
	{
		a = _mm_loadl_epi64((__m128i*)&block2[v*stride+u]);
		b = _mm_loadl_epi64((__m128i*)&block1[v*stride+u]);
		c = _mm_sad_epu8(a, b);
		_mm_storel_epi64((__m128i*)&tempResult, c);

		*result += tempResult;
	}
}
#endif