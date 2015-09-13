#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "tables.h"

#include<stdio.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<x86intrin.h>
#include<pmmintrin.h>
#include<tmmintrin.h>
#include<immintrin.h>
#include<smmintrin.h>

#define ISQRT2 0.70710678118654f

static void transpose_block(float *in_data, float *out_data)
{
  __m128 row0, row1, row2, row3;

  //Load the four rows of the upper left block into registers, transpose them and store them back into memory
  row0 = _mm_load_ps(&in_data[0]);
  row1 = _mm_load_ps(&in_data[8]);
  row2 = _mm_load_ps(&in_data[16]);
  row3 = _mm_load_ps(&in_data[24]);
  _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
  _mm_store_ps(&out_data[0], row0);
  _mm_store_ps(&out_data[8], row1);
  _mm_store_ps(&out_data[16], row2);
  _mm_store_ps(&out_data[24], row3);

  //Load the four rows of the bottom left block into registers, transpose them and store them back into memory
  row0 = _mm_load_ps(&in_data[32]);
  row1 = _mm_load_ps(&in_data[40]);
  row2 = _mm_load_ps(&in_data[48]);
  row3 = _mm_load_ps(&in_data[56]);
  _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
  _mm_store_ps(&out_data[4], row0);
  _mm_store_ps(&out_data[12], row1);
  _mm_store_ps(&out_data[20], row2);
  _mm_store_ps(&out_data[28], row3);

  //Load the four rows of the upper right block into registers, transpose them and store them back into memory
  row0 = _mm_load_ps(&in_data[4]);
  row1 = _mm_load_ps(&in_data[12]);
  row2 = _mm_load_ps(&in_data[20]);
  row3 = _mm_load_ps(&in_data[28]);
  _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
  _mm_store_ps(&out_data[32], row0);
  _mm_store_ps(&out_data[40], row1);
  _mm_store_ps(&out_data[48], row2);
  _mm_store_ps(&out_data[56], row3);

  //Load the four rows of the bottom right block into registers, transpose them and store them back into memory
  row0 = _mm_load_ps(&in_data[36]);
  row1 = _mm_load_ps(&in_data[44]);
  row2 = _mm_load_ps(&in_data[52]);
  row3 = _mm_load_ps(&in_data[60]);
  _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
  _mm_store_ps(&out_data[36], row0);
  _mm_store_ps(&out_data[44], row1);
  _mm_store_ps(&out_data[52], row2);
  _mm_store_ps(&out_data[60], row3);
}

static void dct_1d(float *in_data, float *out_data)
{
  int i, j;
  __m128 indata1, table1, indata2, table2;
  __m128 mulResult1, mulResult2;
  __m128 addResult;

  for (i = 0; i < 8; ++i)
  {
    //Load the first four values of a row into register for data and lookup table
    j = 0;
    indata1 = _mm_load_ps(&in_data[j]);
    table1 = _mm_load_ps(&dctlookupT[i][j]);

    //Load the last four values of a row into register for data and lookup table
    j = 4;
    indata2 = _mm_load_ps(&in_data[j]);
    table2 = _mm_load_ps(&dctlookupT[i][j]);

    //Multiply indata1 with table1 and indata2 with table2
    mulResult1 = _mm_mul_ps(indata1, table1);
    mulResult2 = _mm_mul_ps(indata2, table2);

    //Add the two result with each other and then perform horizontal add to obtain the final value
    addResult = _mm_add_ps(mulResult1, mulResult2);
    addResult = _mm_hadd_ps(addResult, addResult);
    addResult = _mm_hadd_ps(addResult, addResult);

    //Store the final value into out_data
    _mm_store_ss(&out_data[i], addResult);
  }
}

static void idct_1d(float *in_data, float *out_data)
{
  int i, j;
  __m128 indata1, table1, indata2, table2;
  __m128 mulResult1, mulResult2;
  __m128 addResult;

  for (i = 0; i < 8; ++i)
  {
	//Load the first four values of a row into register for data and lookup table
    j = 0;
    indata1 = _mm_load_ps(&in_data[j]);
    table1 = _mm_load_ps(&dctlookup[i][j]);

    //Load the last four values of a row into register for data and lookup table
    j = 4;
    indata2 = _mm_load_ps(&in_data[j]);
    table2 = _mm_load_ps(&dctlookup[i][j]);

    //Multiply indata1 with table1 and indata2 with table2
    mulResult1 = _mm_mul_ps(indata1, table1);
    mulResult2 = _mm_mul_ps(indata2, table2);

    //Add the two result with each other and then perform horizontal add to obtain the final value
    addResult = _mm_add_ps(mulResult1, mulResult2);
    addResult = _mm_hadd_ps(addResult, addResult);
    addResult = _mm_hadd_ps(addResult, addResult);

    //Store the final value into out_data
   _mm_store_ss(&out_data[i], addResult);
  }
}

static void scale_block(float *in_data, float *out_data)
{
  int v;
  __m128 in, tb1, tb2;
  __m128 result;

  for (v = 0; v < 8; ++v)
  {
    int row = v*8;

    //Load in the first four values of a row for in_data, tablea1 and tablea2 to registers
    in = _mm_load_ps(in_data+row);
    tb1 = _mm_load_ps(tablea1+row);
    tb2 = _mm_load_ps(tablea2+row);
    //Multiply in_data with with tablea1 and the result of this with tablea2
    result = _mm_mul_ps(in, tb1);
    result = _mm_mul_ps(result, tb2);
    //Finally store the result to the first four elements of a row in out_data
    _mm_store_ps(out_data+row, result);

    //Load in the last four values of a row for in_data, tablea1 and tablea2 to registers
    in = _mm_load_ps(in_data+row+4);
    tb1 = _mm_load_ps(tablea1+row+4);
    tb2 = _mm_load_ps(tablea2+row+4);
    //Multiply in_data with with tablea1 and the result of this with tablea2
    result = _mm_mul_ps(in, tb1);
    result = _mm_mul_ps(result, tb2);
    //Finally store the result to the last four elements of a row in out_data
    _mm_store_ps(out_data+row+4, result);
  }
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
  int zigzag;

  __m128 dctReg, fourReg, quantReg;
  __m128 result, roundResult;

  //We want to do four operations in parallel, so jump four iterations forward every time
  for (zigzag = 0; zigzag < 64; zigzag+=4)
  {
	  //Create four pair of indexes used to retrieve dct values
    uint8_t u1 = zigzag_U[zigzag];
    uint8_t v1 = zigzag_V[zigzag];

    uint8_t u2 = zigzag_U[zigzag+1];
    uint8_t v2 = zigzag_V[zigzag+1];

    uint8_t u3 = zigzag_U[zigzag+2];
    uint8_t v3 = zigzag_V[zigzag+2];

    uint8_t u4 = zigzag_U[zigzag+3];
    uint8_t v4 = zigzag_V[zigzag+3];

    //Retrieve four dct values
    float dct1 = in_data[v1*8+u1];
    float dct2 = in_data[v2*8+u2];
    float dct3 = in_data[v3*8+u3];
    float dct4 = in_data[v4*8+u4];

    //Insert the four dct values into a register, broadcast the value 4 to each element of a register,
    //and insert four values of the quantization table into a register
    dctReg = _mm_setr_ps(dct1, dct2, dct3, dct4);
    fourReg = _mm_set1_ps(4.0);
    quantReg = _mm_setr_ps((float)quant_tbl[zigzag], (float)quant_tbl[zigzag+1], (float)quant_tbl[zigzag+2], (float)quant_tbl[zigzag+3]);

    //Then we are ready to compute four quantizations in "parallel"
    //We first divide the dct register with the register that has four elements with value 4.0
    result = _mm_div_ps(dctReg, fourReg);
    //We then divide the result of that with the quantization register
    result = _mm_div_ps(result, quantReg);
    //And finally round the four resulting values in parallel and store them in out_data
    roundResult = _mm_round_ps(result, _MM_FROUND_TO_NEAREST_INT);
    _mm_store_ps(&out_data[zigzag], roundResult);
  }
}

static void dequantize_block(float *in_data, float *out_data,
    uint8_t *quant_tbl)
{
  int zigzag;
  float store[4];

  __m128 dctReg, fourReg, quantReg;
  __m128 result, roundResult;

  //We want to do four operations in parallel, so jump four iterations forward every time
  for (zigzag = 0; zigzag < 64; zigzag+=4)
  {
	//Create four pair of indexes used to access out_data when storing back results
    uint8_t u1 = zigzag_U[zigzag];
    uint8_t v1 = zigzag_V[zigzag];

    uint8_t u2 = zigzag_U[zigzag+1];
    uint8_t v2 = zigzag_V[zigzag+1];

    uint8_t u3 = zigzag_U[zigzag+2];
    uint8_t v3 = zigzag_V[zigzag+2];

    uint8_t u4 = zigzag_U[zigzag+3];
    uint8_t v4 = zigzag_V[zigzag+3];

    //Retrieve four dct values
    float dct1 = in_data[zigzag];
    float dct2 = in_data[zigzag+1];
    float dct3 = in_data[zigzag+2];
    float dct4 = in_data[zigzag+3];

    //Insert the four dct values into a register, broadcast the value 4 to each element of a register,
    //and insert four values of the quantization table into a register
    dctReg = _mm_setr_ps(dct1, dct2, dct3, dct4);
    fourReg = _mm_set1_ps(4.0);
    quantReg = _mm_setr_ps((float)quant_tbl[zigzag], (float)quant_tbl[zigzag+1], (float)quant_tbl[zigzag+2], (float)quant_tbl[zigzag+3]);

    //Then we are ready to compute four dequantizations in "parallel"
    //We first multiply the dct register with the quantization register
    result = _mm_mul_ps(dctReg, quantReg);
    //Then we divide that result with the register that has four 4.0 values
    result = _mm_div_ps(result, fourReg);
    //Finally we round four values in parallel and store these to out_data
    roundResult = _mm_round_ps(result, _MM_FROUND_TO_NEAREST_INT);
    _mm_store_ps(&store[0], roundResult);
    out_data[v1*8+u1] = store[0];
    out_data[v2*8+u2] = store[1];
    out_data[v3*8+u3] = store[2];
    out_data[v4*8+u4] = store[3];
  }
}

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

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
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

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
    a_b2 = (__m128i) _mm_loadh_pd((__m128d)b_b2, (double*)&block2[v*stride]);
    a_b1 = (__m128i) _mm_loadh_pd((__m128d)b_b1, (double*)&block1[v*stride]);

    //Compute the absolute difference
    tempResult = _mm_sad_epu8(a_b2, a_b1);
    total = _mm_add_epi16(total, tempResult);
  }
  *result += _mm_extract_epi16(total, 0);
  *result += _mm_extract_epi16(total, 4);
}
