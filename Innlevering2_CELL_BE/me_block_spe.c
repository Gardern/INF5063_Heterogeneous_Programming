#include<stdio.h>
#include<limits.h>
#include<inttypes.h>
#include<math.h>
#include<spu_intrinsics.h>
#include<spu_mfcio.h>

struct macroblock
{
	int use_mv;
	int8_t mv_x, mv_y;
} __attribute__ ((aligned(16)));

#define MB_SIZE 64
#define MAX_REF_SIZE 2048

uint8_t orig_in[2][MB_SIZE] __attribute__ ((aligned(128)));
uint8_t ref_in[2][MAX_REF_SIZE] __attribute__ ((aligned(16)));
struct macroblock mb_out __attribute__ ((aligned(16)));
unsigned long long addressesY[2048] __attribute__ ((aligned(16)));
unsigned long long addressesU[512] __attribute__ ((aligned(16)));
unsigned long long addressesV[512] __attribute__ ((aligned(16)));
int tag;

/*
//MACROS FOR DEBUGGING DMAs
#define spu_mfcdma32(ls, l, sz, tag, cmd){ \
    printf("spu_mfcdma32(%p, %x, %d, %d, %d) -- Line: %d\n", ls, l, sz, tag, cmd, __LINE__); \
    spu_mfcdma32(ls, l, sz, tag, cmd); \
}
#define spu_mfcdma64(ls, h, l, sz, tag, cmd){ \
    printf("spu_mfcdma64(%p, %x, %x, %d, %d, %d) -- Line: %d\n", ls, h, l, sz, tag, cmd, __LINE__); \
    spu_mfcdma64(ls, h, l, sz, tag, cmd); \
}
*/

typedef struct
{
	unsigned long long ea_origY_data_in;
	unsigned long long ea_origU_data_in;
	unsigned long long ea_origV_data_in;
	unsigned long long ea_reconsY_data_in;
	unsigned long long ea_reconsU_data_in;
	unsigned long long ea_reconsV_data_in;
	unsigned long long ea_mbY_out;
	unsigned long long ea_mbU_out;
	unsigned long long ea_mbV_out;
	int mb_x;
	int mb_y_Y;
	int mb_y_UV;
	int mb_max_y_Y;
	int mb_max_y_UV;
	int mb_cols;
	int mb_rows;
	vector float range;
	int padYw;
	int padYh;
	int padUw;
	int padUh;
	int padVw;
	int padVh;
} me_parm_data __attribute__ ((aligned (128)));

me_parm_data spe_me_data;

typedef struct
{
	vector float lt;
	vector float rb;
	int macroblock;
	vector float hv_range;
	vector float eight;
} motion_estimation_info;

motion_estimation_info me_info;

vector unsigned char a1, a2, d;
vector unsigned short s;

//Motion estimation for 8x8 block
void new_me_block_8x8(vector float mb_xy, vector float lt, vector float rb, vector float hv_range, uint8_t *orig, uint8_t *ref) //struct c63_common *cm, int mb_x, int mb_y, uint8_t *orig, uint8_t *ref, int cc
{
	mb_out.mv_x = 0;
	mb_out.mv_y = 0;
	mb_out.use_mv = 0;

	int x, y;
	int newX, newY;

	vector float mxy = spu_mul(mb_xy, me_info.eight);

	int best_sad = INT_MAX;

	newY = 0;
	for (y = lt[1]; y < rb[1]; ++y, ++newY)
	{
		newX = 0;
		for (x = lt[0]; x < rb[0]; ++x, ++newX)
		{
			int sad = 0;
			vector float whole_hv_range = spu_add(hv_range, me_info.eight);

			uint8_t *newRef = ref + newY*((int)whole_hv_range[0])+newX;
			int stride = (int)whole_hv_range[0];

			int v, v2;
			int i = 0;

			vector unsigned char tempChars = (vector unsigned char){0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
			vector unsigned short tempShorts = (vector unsigned short){0,0,0,0,0,0,0,0};

			for (v = 0, v2 = 1; v < 8; v+=2, v2+=2)
			{
				a1 = (vector unsigned char) {orig[i+0], orig[i+1], orig[i+2], orig[i+3],
											orig[i+4], orig[i+5], orig[i+6], orig[i+7],
											orig[i+8], orig[i+9], orig[i+10], orig[i+11],
											orig[i+12], orig[i+13], orig[i+14], orig[i+15]};

				a2 = (vector unsigned char) {newRef[v*stride+0], newRef[v*stride+1], newRef[v*stride+2], newRef[v*stride+3],
											newRef[v*stride+4], newRef[v*stride+5], newRef[v*stride+6], newRef[v*stride+7],
											newRef[v2*stride+0], newRef[v2*stride+1], newRef[v2*stride+2], newRef[v2*stride+3],
											newRef[v2*stride+4], newRef[v2*stride+5], newRef[v2*stride+6], newRef[v2*stride+7]};
				d = spu_absd(a2, a1);
				s = spu_sumb(d, tempChars);
				tempShorts = spu_add(s, tempShorts);
				i+=16;
			}
			int temp = tempShorts[0] + tempShorts[1] + tempShorts[2] + tempShorts[3] +
						tempShorts[4] + tempShorts[5] + tempShorts[6] + tempShorts[7];
			sad += temp;

			if (sad < best_sad)
			{
				mb_out.mv_x = x - mxy[0];
				mb_out.mv_y = y - mxy[1];
				best_sad = sad;
			}
		}
	}

	// Here, there should be a threshold on SAD that checks if the motion vector is cheaper than intraprediction. We always assume MV to be beneficial
	// printf("Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y, best_sad);

	mb_out.use_mv = 1;
}

int main(unsigned long long spe, unsigned long long argp, unsigned long long envp)
{
	tag = mfc_tag_reserve();
	if(tag == MFC_TAG_INVALID)
	{
		printf("%s \n", "SPE: Cant allocate tag");
	}

	spu_mfcdma32(&spe_me_data, argp, sizeof(spe_me_data), tag, MFC_GET_CMD);
	spu_writech(MFC_WrTagMask, 1 << tag);
	spu_mfcstat(MFC_TAG_UPDATE_ALL);

	spu_mfcdma32(addressesY, spe_me_data.ea_reconsY_data_in, (spe_me_data.mb_rows * spe_me_data.mb_cols) * sizeof(unsigned long long), tag, MFC_GET_CMD);
	spu_mfcdma32(addressesU, spe_me_data.ea_reconsU_data_in, ((spe_me_data.mb_rows * spe_me_data.mb_cols) / 4) * sizeof(unsigned long long), tag, MFC_GET_CMD);;
	spu_mfcdma32(addressesV, spe_me_data.ea_reconsV_data_in, ((spe_me_data.mb_rows * spe_me_data.mb_cols) / 4) * sizeof(unsigned long long), tag, MFC_GET_CMD);
	spu_writech(MFC_WrTagMask, 1 << tag);
	spu_mfcstat(MFC_TAG_UPDATE_ALL);

	int next_idx, buf_idx;
	vector float mb_xy = (vector float){0, 0, 0, 0}; //x, y
	vector float previous_mb_xy = (vector float){0, 0, 0, 0};
	vector float previous_lt = (vector float){0, 0, 0, 0};
	vector float previous_rb = (vector float){0, 0, 0, 0};
	vector float previous_hv_range = (vector float){0, 0, 0, 0};
	me_info.eight = spu_splats((float)8);
	vector float lt_border = spu_splats((float)0);
	vector float test_value = spu_splats((float)-1);
	vector float rb_border = (vector float){spe_me_data.padYw - 8, spe_me_data.padYh - 8, 0, 0};

	mb_xy[1] = spe_me_data.mb_y_Y;
	mb_xy[0] = spe_me_data.mb_x;

	previous_mb_xy = mb_xy;

	int t = 1;

	me_info.macroblock = mb_xy[1] * spe_me_data.padYw/8 + mb_xy[0];
	me_info.lt = spu_msub(mb_xy, me_info.eight, spe_me_data.range);
	me_info.rb = spu_madd(mb_xy, me_info.eight, spe_me_data.range);

	vector unsigned int cmp_result;
	cmp_result = spu_cmpgt(me_info.lt, test_value);
	me_info.lt = spu_sel(lt_border, me_info.lt, cmp_result);
	cmp_result = spu_cmpgt(me_info.rb, rb_border);
	me_info.rb = spu_sel(me_info.rb, rb_border, cmp_result);

	me_info.hv_range = spu_sub(me_info.rb, me_info.lt);

	previous_lt = me_info.lt;
	previous_rb = me_info.rb;
	previous_hv_range = me_info.hv_range;

	int nextOrigBlock = me_info.macroblock * 64;
	int REF_SIZE = ((int)me_info.hv_range[0] + 8) * ((int)me_info.hv_range[1] + 8);

	next_idx, buf_idx = 0;


	//Start first DMA transfer
	spu_mfcdma32(orig_in[buf_idx], spe_me_data.ea_origY_data_in + nextOrigBlock, MB_SIZE * sizeof(uint8_t), buf_idx, MFC_GET_CMD);
	spu_mfcdma32(ref_in[buf_idx], addressesY[me_info.macroblock], REF_SIZE, buf_idx, MFC_GET_CMD);
	//Luma
	for (mb_xy[1] = spe_me_data.mb_y_Y; mb_xy[1] < spe_me_data.mb_max_y_Y; ++mb_xy[1])
	{
		for (mb_xy[0] = spe_me_data.mb_x + t; mb_xy[0] < spe_me_data.mb_cols; ++mb_xy[0])
		{
			me_info.macroblock = mb_xy[1] * spe_me_data.padYw/8 + mb_xy[0];
			me_info.lt = spu_msub(mb_xy, me_info.eight, spe_me_data.range);
			me_info.rb = spu_madd(mb_xy, me_info.eight, spe_me_data.range);

			cmp_result = spu_cmpgt(me_info.lt, test_value);
			me_info.lt = spu_sel(lt_border, me_info.lt, cmp_result);
			cmp_result = spu_cmpgt(me_info.rb, rb_border);
			me_info.rb = spu_sel(me_info.rb, rb_border, cmp_result);

			me_info.hv_range = spu_sub(me_info.rb, me_info.lt);

			int nextOrigBlock = me_info.macroblock * 64;
			int REF_SIZE = ((int)me_info.hv_range[0] + 8) * ((int)me_info.hv_range[1] + 8);

			next_idx = buf_idx ^ 1;

			//Start next DMA transfer
			spu_mfcdma32(orig_in[next_idx], spe_me_data.ea_origY_data_in + nextOrigBlock, MB_SIZE * sizeof(uint8_t), next_idx, MFC_GET_CMD);
			spu_mfcdma32(ref_in[next_idx], addressesY[me_info.macroblock], REF_SIZE, next_idx, MFC_GET_CMD);

			//Wait for previous transfer
			spu_writech(MFC_WrTagMask, 1 << buf_idx);
			spu_mfcstat(MFC_TAG_UPDATE_ALL);

			//Process the data from previous transfer
			new_me_block_8x8(previous_mb_xy, previous_lt, previous_rb, previous_hv_range, orig_in[buf_idx], ref_in[buf_idx]);

			previous_mb_xy = mb_xy;
			previous_lt = me_info.lt;
			previous_rb = me_info.rb;
			previous_hv_range = me_info.hv_range;

			spu_mfcdma32(&mb_out, spe_me_data.ea_mbY_out+(sizeof(struct macroblock) * (me_info.macroblock - 1)), sizeof(struct macroblock), buf_idx, MFC_PUT_CMD);
			spu_writech(MFC_WrTagMask, 1 << buf_idx);
			spu_mfcstat(MFC_TAG_UPDATE_ALL);

			buf_idx = next_idx;
		}
		t = 0;
	}

	//Wait for last transfer
	spu_writech(MFC_WrTagMask, 1 << buf_idx);
	spu_mfcstat(MFC_TAG_UPDATE_ALL);

	//Process the data from last transfer
	new_me_block_8x8(previous_mb_xy, previous_lt, previous_rb, previous_hv_range, orig_in[buf_idx], ref_in[buf_idx]);

	spu_mfcdma32(&mb_out, spe_me_data.ea_mbY_out+(sizeof(struct macroblock) * me_info.macroblock), sizeof(struct macroblock), buf_idx, MFC_PUT_CMD);
	spu_writech(MFC_WrTagMask, 1 << buf_idx);
	spu_mfcstat(MFC_TAG_UPDATE_ALL);

;
	vector float half = spu_splats((float)0.5);
	spe_me_data.range = spu_mul(spe_me_data.range, half);
	rb_border = (vector float){spe_me_data.padUw - 8, spe_me_data.padUh - 8, 0, 0};
	int cols = spe_me_data.mb_cols * 0.5f;
	//Chroma
	for (mb_xy[1] = spe_me_data.mb_y_UV; mb_xy[1] < spe_me_data.mb_max_y_UV; ++mb_xy[1])
	{
		for (mb_xy[0] = spe_me_data.mb_x; mb_xy[0] < cols; ++mb_xy[0])
		{
			me_info.macroblock = mb_xy[1] * spe_me_data.padUw/8 + mb_xy[0];

			me_info.lt = spu_msub(mb_xy, me_info.eight, spe_me_data.range);
			me_info.rb = spu_madd(mb_xy, me_info.eight, spe_me_data.range);

			vector unsigned int cmp_result;
			cmp_result = spu_cmpgt(me_info.lt, test_value);
			me_info.lt = spu_sel(lt_border, me_info.lt, cmp_result);
			cmp_result = spu_cmpgt(me_info.rb, rb_border);
			me_info.rb = spu_sel(me_info.rb, rb_border, cmp_result);

			me_info.hv_range = spu_sub(me_info.rb, me_info.lt);

			int nextOrigBlock = me_info.macroblock * 64;
			int REF_SIZE = ((int)me_info.hv_range[0] + 8) * ((int)me_info.hv_range[1] + 8);

			spu_mfcdma32(orig_in, spe_me_data.ea_origU_data_in + nextOrigBlock, MB_SIZE * sizeof(uint8_t), tag, MFC_GET_CMD);
			spu_writech(MFC_WrTagMask, 1 << tag);
			spu_mfcstat(MFC_TAG_UPDATE_ALL);

			spu_mfcdma32(ref_in, addressesU[me_info.macroblock], REF_SIZE, tag, MFC_GET_CMD);
			spu_writech(MFC_WrTagMask, 1 << tag);
			spu_mfcstat(MFC_TAG_UPDATE_ALL);

			new_me_block_8x8(mb_xy, me_info.lt, me_info.rb, me_info.hv_range, orig_in, ref_in);

			spu_mfcdma32(&mb_out, spe_me_data.ea_mbU_out+(sizeof(struct macroblock) * me_info.macroblock), sizeof(struct macroblock), tag, MFC_PUT_CMD);
			spu_writech(MFC_WrTagMask, 1 << tag);
			spu_mfcstat(MFC_TAG_UPDATE_ALL);


			spu_mfcdma32(orig_in, spe_me_data.ea_origV_data_in + nextOrigBlock, MB_SIZE * sizeof(uint8_t), tag, MFC_GET_CMD);
			spu_writech(MFC_WrTagMask, 1 << tag);
			spu_mfcstat(MFC_TAG_UPDATE_ALL);

			spu_mfcdma32(ref_in, addressesV[me_info.macroblock], REF_SIZE, tag, MFC_GET_CMD);
			spu_writech(MFC_WrTagMask, 1 << tag);
			spu_mfcstat(MFC_TAG_UPDATE_ALL);

			new_me_block_8x8(mb_xy, me_info.lt, me_info.rb, me_info.hv_range, orig_in, ref_in);

			spu_mfcdma32(&mb_out, spe_me_data.ea_mbV_out+(sizeof(struct macroblock) * me_info.macroblock), sizeof(struct macroblock), tag, MFC_PUT_CMD);
			spu_writech(MFC_WrTagMask, 1 << tag);
			spu_mfcstat(MFC_TAG_UPDATE_ALL);
		}
	}

	return 0;
}
