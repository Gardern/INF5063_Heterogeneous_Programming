#include<stdio.h>
#include<limits.h>
#include<inttypes.h>
#include<math.h>
#include <stdlib.h>

#include<spu_intrinsics.h>
#include<spu_mfcio.h>

#include<unistd.h>
#include <simdmath.h>

#define ISQRT2 0.70710678118654f

int16_t in_data[44*64] __attribute__ ((aligned(16)));
uint8_t prediction[44*64] __attribute__ ((aligned(16)));
uint8_t quantization[64] __attribute__ ((aligned(16)));
uint8_t out_data[44*64] __attribute__ ((aligned(16)));

typedef struct
{
	unsigned long long in_data;
	unsigned long long prediction;
	int width;
	int height;
	int y;
	int range;
	unsigned long long out_data;
	unsigned long long quantization;
} idct_parm_data;

idct_parm_data spe_idct_data __attribute__ ((aligned (16)));

vector unsigned char high = ((vector unsigned char) {
	0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17});


vector unsigned char low = ((vector unsigned char) {
	0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B, 0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F});

uint8_t zigzag_U[64] =
{
  0,
  1, 0,
  0, 1, 2,
  3, 2, 1, 0,
  0, 1, 2, 3, 4,
  5, 4, 3, 2, 1, 0,
  0, 1, 2, 3, 4, 5, 6,
  7, 6, 5, 4, 3, 2, 1, 0,
  1, 2, 3, 4, 5, 6, 7,
  7, 6, 5, 4, 3, 2,
  3, 4, 5, 6, 7,
  7, 6, 5, 4,
  5, 6, 7,
  7, 6,
  7,
};

uint8_t zigzag_V[64] =
{
  0,
  0, 1,
  2, 1, 0,
  0, 1, 2, 3,
  4, 3, 2, 1, 0,
  0, 1, 2, 3, 4, 5,
  6, 5, 4, 3, 2, 1, 0,
  0, 1, 2, 3, 4, 5, 6, 7,
  7, 6, 5, 4, 3, 2, 1,
  2, 3, 4, 5, 6, 7,
  7, 6, 5, 4, 3,
  4, 5, 6, 7,
  7, 6, 5,
  6, 7,
  7,
};

float dctlookup[8][8] =
{
  {1.0f,  0.980785f,  0.923880f,  0.831470f,  0.707107f,  0.555570f,  0.382683f,  0.195090f, },
  {1.0f,  0.831470f,  0.382683f, -0.195090f, -0.707107f, -0.980785f, -0.923880f, -0.555570f, },
  {1.0f,  0.555570f, -0.382683f, -0.980785f, -0.707107f,  0.195090f,  0.923880f,  0.831470f, },
  {1.0f,  0.195090f, -0.923880f, -0.555570f,  0.707107f,  0.831470f, -0.382683f, -0.980785f, },
  {1.0f, -0.195090f, -0.923880f,  0.555570f,  0.707107f, -0.831470f, -0.382683f,  0.980785f, },
  {1.0f, -0.555570f, -0.382683f,  0.980785f, -0.707107f, -0.195090f,  0.923880f, -0.831470f, },
  {1.0f, -0.831470f,  0.382683f,  0.195090f, -0.707107f,  0.980785f, -0.923880f,  0.555570f, },
  {1.0f, -0.980785f,  0.923880f, -0.831470f,  0.707107f, -0.555570f,  0.382683f, -0.195090f, },
};

float tablea1[64] __attribute__((aligned(16))) =
{
	0.70710678118654f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
	0.70710678118654f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
	0.70710678118654f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
	0.70710678118654f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
	0.70710678118654f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
	0.70710678118654f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
	0.70710678118654f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
	0.70710678118654f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f
};

float tablea2[64] __attribute__((aligned(16))) =
{
	0.70710678118654f, 0.70710678118654f, 0.70710678118654f, 0.70710678118654f, 0.70710678118654f, 0.70710678118654f, 0.70710678118654f, 0.70710678118654f,
	1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
	1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
	1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
	1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
	1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
	1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
	1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f
};

/*
#define spu_mfcdma32(ls, l, sz, tag, cmd) { \
    printf("spu_mfcdma32(%p, %x, %d, %d, %d) -- Line: %d\n", ls, l, sz, tag, cmd, __LINE__); \
    spu_mfcdma32(ls, l, sz, tag, cmd); \
}

#define spu_mfcdma64(ls, h, l, sz, tag, cmd) { \
    printf("spu_mfcdma64(%p, %x, %x, %d, %d, %d) -- Line: %d\n", ls, h, l, sz, tag, cmd, __LINE__); \
    spu_mfcdma64(ls, h, l, sz, tag, cmd); \
}
*/

static void transpose_4x4_block(vector float *row0, vector float *row1, vector float *row2, vector float *row3)
{
	vector float temp0, temp1, temp2, temp3;

	temp0 = spu_shuffle(*row0, *row2, high);
	temp1 = spu_shuffle(*row0, *row2, low);
	temp2 = spu_shuffle(*row1, *row3, high);
	temp3 = spu_shuffle(*row1, *row3, low);

	*row0 = spu_shuffle(temp0, temp2, high);
	*row1 = spu_shuffle(temp0, temp2, low);
	*row2 = spu_shuffle(temp1, temp3, high);
	*row3 = spu_shuffle(temp1, temp3, low);
}

static void transpose_block(float *in_data, float *out_data)
{
	vector float row0, row1, row2, row3;

	row0 = (vector float) { *(in_data+0), *(in_data+1), *(in_data+2), *(in_data+3) };
	row1 = (vector float) { *(in_data+8), *(in_data+9), *(in_data+10), *(in_data+11) };
	row2 = (vector float) { *(in_data+16), *(in_data+17), *(in_data+18), *(in_data+19) };
	row3 = (vector float) { *(in_data+24), *(in_data+25), *(in_data+26), *(in_data+27) };
	transpose_4x4_block(&row0, &row1, &row2, &row3);
	*(out_data+0) = spu_extract(row0, 0), *(out_data+1) = spu_extract(row0, 1), *(out_data+2) = spu_extract(row0, 2), *(out_data+3) = spu_extract(row0, 3);
	*(out_data+8) = spu_extract(row1, 0), *(out_data+9) = spu_extract(row1, 1), *(out_data+10) = spu_extract(row1, 2), *(out_data+11) = spu_extract(row1, 3);
	*(out_data+16) = spu_extract(row2, 0), *(out_data+17) = spu_extract(row2, 1), *(out_data+18) = spu_extract(row2, 2), *(out_data+19) = spu_extract(row2, 3);
	*(out_data+24) = spu_extract(row3, 0), *(out_data+25) = spu_extract(row3, 1), *(out_data+26) = spu_extract(row3, 2), *(out_data+27) = spu_extract(row3, 3);

	row0 = (vector float) { *(in_data+32), *(in_data+33), *(in_data+34), *(in_data+35) };
	row1 = (vector float) { *(in_data+40), *(in_data+41), *(in_data+42), *(in_data+43) };
	row2 = (vector float) { *(in_data+48), *(in_data+49), *(in_data+50), *(in_data+51) };
	row3 = (vector float) { *(in_data+56), *(in_data+57), *(in_data+58), *(in_data+59) };
	transpose_4x4_block(&row0, &row1, &row2, &row3);
	*(out_data+4) = spu_extract(row0, 0), *(out_data+5) = spu_extract(row0, 1), *(out_data+6) = spu_extract(row0, 2), *(out_data+7) = spu_extract(row0, 3);
	*(out_data+12) = spu_extract(row1, 0), *(out_data+13) = spu_extract(row1, 1), *(out_data+14) = spu_extract(row1, 2), *(out_data+15) = spu_extract(row1, 3);
	*(out_data+20) = spu_extract(row2, 0), *(out_data+21) = spu_extract(row2, 1), *(out_data+22) = spu_extract(row2, 2), *(out_data+23) = spu_extract(row2, 3);
	*(out_data+28) = spu_extract(row3, 0), *(out_data+29) = spu_extract(row3, 1), *(out_data+30) = spu_extract(row3, 2), *(out_data+31) = spu_extract(row3, 3);

	row0 = (vector float) { *(in_data+4), *(in_data+5), *(in_data+6), *(in_data+7) };
	row1 = (vector float) { *(in_data+12), *(in_data+13), *(in_data+14), *(in_data+15) };
	row2 = (vector float) { *(in_data+20), *(in_data+21), *(in_data+22), *(in_data+23) };
	row3 = (vector float) { *(in_data+28), *(in_data+29), *(in_data+30), *(in_data+31) };
	transpose_4x4_block(&row0, &row1, &row2, &row3);
	*(out_data+32) = spu_extract(row0, 0), *(out_data+33) = spu_extract(row0, 1), *(out_data+34) = spu_extract(row0, 2), *(out_data+35) = spu_extract(row0, 3);
	*(out_data+40) = spu_extract(row1, 0), *(out_data+41) = spu_extract(row1, 1), *(out_data+42) = spu_extract(row1, 2), *(out_data+43) = spu_extract(row1, 3);
	*(out_data+48) = spu_extract(row2, 0), *(out_data+49) = spu_extract(row2, 1), *(out_data+50) = spu_extract(row2, 2), *(out_data+51) = spu_extract(row2, 3);
	*(out_data+56) = spu_extract(row3, 0), *(out_data+57) = spu_extract(row3, 1), *(out_data+58) = spu_extract(row3, 2), *(out_data+59) = spu_extract(row3, 3);

	row0 = (vector float) { *(in_data+36), *(in_data+37), *(in_data+38), *(in_data+39) };
	row1 = (vector float) { *(in_data+44), *(in_data+45), *(in_data+46), *(in_data+47) };
	row2 = (vector float) { *(in_data+52), *(in_data+53), *(in_data+54), *(in_data+55) };
	row3 = (vector float) { *(in_data+60), *(in_data+61), *(in_data+62), *(in_data+63) };
	transpose_4x4_block(&row0, &row1, &row2, &row3);
	*(out_data+36) = spu_extract(row0, 0), *(out_data+37) = spu_extract(row0, 1), *(out_data+38) = spu_extract(row0, 2), *(out_data+39) = spu_extract(row0, 3);
	*(out_data+44) = spu_extract(row1, 0), *(out_data+45) = spu_extract(row1, 1), *(out_data+46) = spu_extract(row1, 2), *(out_data+47) = spu_extract(row1, 3);
	*(out_data+52) = spu_extract(row2, 0), *(out_data+53) = spu_extract(row2, 1), *(out_data+54) = spu_extract(row2, 2), *(out_data+55) = spu_extract(row2, 3);
	*(out_data+60) = spu_extract(row3, 0), *(out_data+61) = spu_extract(row3, 1), *(out_data+62) = spu_extract(row3, 2), *(out_data+63) = spu_extract(row3, 3);
}

static void idct_1d(float *in_data, float *out_data)
{
	int i;

	for (i = 0; i < 8; ++i)
	{
		vector float tempResult = (vector float) {0.0f, 0.0f, 0.0f, 0.0f};

		vector float in_data1 = (vector float) {in_data[0], in_data[1], in_data[2], in_data[3]};
		vector float table1 = (vector float) {dctlookup[i][0], dctlookup[i][1], dctlookup[i][2], dctlookup[i][3]};

		tempResult = spu_madd(in_data1, table1, tempResult);

		vector float in_data2 = (vector float) {in_data[4], in_data[5], in_data[6], in_data[7]};
		vector float table2 = (vector float ) {dctlookup[i][4], dctlookup[i][5], dctlookup[i][6], dctlookup[i][7]};

		tempResult = spu_madd(in_data1, table1, tempResult);

		out_data[i] = tempResult[0] + tempResult[1] + tempResult[2] + tempResult[3];
	}
}

static void scale_block(float *in_data, float *out_data)
{
	int i;

	vector float *in = (vector float *) (in_data);
	vector float *tb1 = (vector float *) (tablea1);
	vector float *tb2 = (vector float *) (tablea2);
	vector float *out = (vector float *) (out_data);

	for (i = 0; i < 16; ++i)
	{
		out[i] = spu_mul(in[i], tb1[i]);
		out[i] = spu_mul(out[i], tb2[i]);
	}
}

static void dequantize_block(float *in_data, float *out_data,
    uint8_t *quant_tbl)
{
	int zigzag;

	vector float zero_p_twof = spu_splats(0.25f);

	for (zigzag = 0; zigzag < 64;)
	{
		uint8_t u1 = zigzag_U[zigzag];
		uint8_t v1 = zigzag_V[zigzag];

		uint8_t u2 = zigzag_U[zigzag+1];
		uint8_t v2 = zigzag_V[zigzag+1];

		uint8_t u3 = zigzag_U[zigzag+2];
		uint8_t v3 = zigzag_V[zigzag+2];

		uint8_t u4 = zigzag_U[zigzag+3];
		uint8_t v4 = zigzag_V[zigzag+3];

		float dct1 = in_data[zigzag];
		float dct2 = in_data[zigzag+1];
		float dct3 = in_data[zigzag+2];
		float dct4 = in_data[zigzag+3];

		vector float dct = (vector float) {dct1, dct2, dct3, dct4};
		vector float quant = (vector float) {quant_tbl[zigzag], quant_tbl[zigzag+1], quant_tbl[zigzag+2], quant_tbl[zigzag+3]};

		vector float result1 = spu_mul(dct, quant);
		result1 = spu_mul(result1, zero_p_twof);

		out_data[v1*8+u1] = round(result1[0]);
		out_data[v2*8+u2] = round(result1[1]);
		out_data[v3*8+u3] = round(result1[2]);
		out_data[v4*8+u4] = round(result1[3]);

		zigzag+=4;
	}
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

void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, int h,
		uint8_t *out_data, uint8_t *quantization)
{
  int x;

  int16_t block[8*8];

  // Perform the dequantization and iDCT
  for(x = 0; x < w/8*64; x += 64)
  {
    int i, j;
    int k;

    dequant_idct_block_8x8(in_data+x, block, quantization); //in_data+(x*8)

    for (i = 0; i < 64; ++i)
    {
		// Add prediction block. Note: DCT is not precise -
		// Clamp to legal values
		int16_t tmp = block[i] + (int16_t)prediction[x+i];

		if (tmp < 0) { tmp = 0; }
		else if (tmp > 255) { tmp = 255; }

		out_data[x+i] = tmp;
    }
  }
}

int main(unsigned long long spe, unsigned long long argp, unsigned long long envp)
{
	int tag = 1;

	spu_mfcdma32(&spe_idct_data, argp, sizeof(spe_idct_data), tag, MFC_GET_CMD);
	spu_writech(MFC_WrTagMask, 1 << tag);
	spu_mfcstat(MFC_TAG_UPDATE_ALL);

	spu_mfcdma32((void*) quantization, (unsigned int)spe_idct_data.quantization, 64 * sizeof(uint8_t), tag, MFC_GET_CMD);

	int range = spe_idct_data.range;

	int i;
  	for (i = spe_idct_data.y; i < range; ++i)
  	{
		spu_mfcdma32((void*) in_data, (unsigned int)spe_idct_data.in_data+(i*((spe_idct_data.width/8)*64))*sizeof(int16_t), ((spe_idct_data.width/8))*64 * sizeof(int16_t), tag, MFC_GET_CMD);
		spu_mfcdma32((void*) prediction, (unsigned int)spe_idct_data.prediction+i*((spe_idct_data.width/8)*64), (spe_idct_data.width/8)*64, tag, MFC_GET_CMD);
		spu_writech(MFC_WrTagMask, 1 << tag);
		spu_mfcstat(MFC_TAG_UPDATE_ALL);

		dequantize_idct_row(in_data, prediction, spe_idct_data.width, spe_idct_data.height,
    			out_data, quantization);

		spu_mfcdma32((void*) out_data, (unsigned int)spe_idct_data.out_data+i*((spe_idct_data.width/8)*64), (spe_idct_data.width/8)*64, tag, MFC_PUT_CMD);
		spu_writech(MFC_WrTagMask, 1 << tag);
		spu_mfcstat(MFC_TAG_UPDATE_ALL);
  	}
}
