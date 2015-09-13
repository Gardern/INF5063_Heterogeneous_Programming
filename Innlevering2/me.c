#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>
#include <libspe2.h>
#include <altivec.h>

#include "c63.h"

#define MAX_SPE_THREADS 6

typedef struct ppu_pthread_data
{
	spe_context_ptr_t spe_ctx;
	spe_program_handle_t *prog;
	pthread_t pthread;
	void *argp;
} ppe_pthread_data_t;

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

me_parm_data ppe_me_data[MAX_SPE_THREADS];
ppe_pthread_data_t data[MAX_SPE_THREADS];

void *recons_data_void __attribute__ ((aligned(16)));
unsigned long long* addressesY __attribute__ ((aligned(16)));
unsigned long long* addressesU __attribute__ ((aligned(16)));
unsigned long long* addressesV __attribute__ ((aligned(16)));

void *ppu_pthread_function(void *arg)
{
	ppe_pthread_data_t *datap = (ppe_pthread_data_t *)arg;
	unsigned int entry = SPE_DEFAULT_ENTRY;

	if(spe_context_run(datap->spe_ctx, &entry, 0, datap->argp, NULL, NULL) < 0)
	{
		perror ("Failed to run context");
		sleep(1000);
	}
	pthread_exit(NULL);
}

//Function for allocating all the necessary data for me on SPEs
void c63_motion_estimate_alloc(struct c63_common *cm)
{
	int mb_x, mb_y;

	posix_memalign((void**)&YreconsData, 16, (cm->mb_rows * cm->mb_cols) * sizeof(uint8_t*));
	posix_memalign((void**)&UreconsData, 16, (cm->mb_rows / 2) * (cm->mb_cols / 2) * sizeof(uint8_t*));
	posix_memalign((void**)&VreconsData, 16, (cm->mb_rows / 2) * (cm->mb_cols / 2) * sizeof(uint8_t*));

	addressesY = (unsigned long long*)malloc((cm->mb_rows * cm->mb_cols) * sizeof(unsigned long long));
	addressesU = (unsigned long long*)malloc((cm->mb_rows / 2) * (cm->mb_cols / 2) * sizeof(unsigned long long));
	addressesV = (unsigned long long*)malloc((cm->mb_rows / 2) * (cm->mb_cols / 2) * sizeof(unsigned long long));

	//Luma
	for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
		{
			int range = cm->me_search_range;
			int cc = 0;
			int macroblock = mb_y * cm->padw[cc]/8 + mb_x;

			if (cc > 0) { range /= 2; }

			int left = mb_x * 8 - range;
			int top = mb_y * 8 - range;
			int right = mb_x * 8 + range;
			int bottom = mb_y * 8 + range;

			int w = cm->padw[cc];
			int h = cm->padh[cc];

			/* Make sure we are within bounds of reference frame. TODO: Support partial
			frame bounds. */
			if (left < 0) { left = 0; }
			if (top < 0) { top = 0; }
			if (right > (w - 8)) { right = w - 8; }
			if (bottom > (h - 8)) { bottom = h - 8; }

			int x, y;
			int newY;

			int horizontalRange = (right - left);
			int verticalRange = (bottom - top);

			int allocate = (horizontalRange + 8) * (verticalRange + 8);

			posix_memalign((void**)&YreconsData[macroblock], 16, allocate * sizeof(uint8_t));

			newY = 0;
			x = left;
			for (y = top; y < bottom + 8; ++y, ++newY)
			{
				memcpy((void*)&YreconsData[macroblock][newY*(horizontalRange + 8)], (void*)&cm->refframe->newRecons->Y[y*w+x], (horizontalRange + 8) * sizeof(uint8_t));
			}
		}
	}
	//Chroma
	for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
		{
			int range = cm->me_search_range;
			int cc = 1;
			int macroblock = mb_y * cm->padw[cc]/8 + mb_x;

			if (cc > 0) { range /= 2; }

			int left = mb_x * 8 - range;
			int top = mb_y * 8 - range;
			int right = mb_x * 8 + range;
			int bottom = mb_y * 8 + range;

			int w = cm->padw[cc];
			int h = cm->padh[cc];

			/* Make sure we are within bounds of reference frame. TODO: Support partial
			frame bounds. */
			if (left < 0) { left = 0; }
			if (top < 0) { top = 0; }
			if (right > (w - 8)) { right = w - 8; }
			if (bottom > (h - 8)) { bottom = h - 8; }

			int x, y;
			int newY;

			int horizontalRange = (right - left);
			int verticalRange = (bottom - top);

			int allocate = (horizontalRange + 8) * (verticalRange + 8);

			posix_memalign((void**)&UreconsData[macroblock], 16, allocate * sizeof(uint8_t));
			posix_memalign((void**)&VreconsData[macroblock], 16, allocate * sizeof(uint8_t));

			newY = 0;
			x = left;
			for (y = top; y < bottom + 8; ++y, ++newY)
			{
				memcpy((void*)&UreconsData[macroblock][newY*(horizontalRange + 8)], (void*)&cm->refframe->newRecons->U[y*w+x], (horizontalRange + 8) * sizeof(uint8_t));
				memcpy((void*)&VreconsData[macroblock][newY*(horizontalRange + 8)], (void*)&cm->refframe->newRecons->V[y*w+x], (horizontalRange + 8) * sizeof(uint8_t));
			}
		}
	}

	int i;
	for(i = 0; i < (cm->mb_rows * cm->mb_cols); ++i)
	{
		addressesY[i] = (unsigned long long) YreconsData[i];
	}
	for(i = 0; i < (cm->mb_rows * cm->mb_cols) / 4; ++i)
	{
		addressesU[i] = (unsigned long long) UreconsData[i];
		addressesV[i] = (unsigned long long) VreconsData[i];
	}
}

//Function for deallocating all the data for me on SPEs
void c63_motion_estimate_dealloc(struct c63_common *cm)
{
	int mb_x, mb_y;

	// Luma
	for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
		{
			int macroblockY = mb_y * cm->padw[0]/8 + mb_x;

			free(YreconsData[macroblockY]);
		}

	}

	// Chroma
	for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
		{
			int macroblockU = mb_y * cm->padw[1]/8 + mb_x;
			int macroblockV = mb_y * cm->padw[2]/8 + mb_x;

			free(UreconsData[macroblockU]);
			free(VreconsData[macroblockV]);
		}
	}

	free(YreconsData);
	free(UreconsData);
	free(VreconsData);

	free(addressesY);
	free(addressesU);
	free(addressesV);
}

void c63_motion_estimate(struct c63_common *cm)
{
	int i;

	//Get the number of SPEs
	int num_spe_threads = spe_cpu_info_get(SPE_COUNT_USABLE_SPES, -1);
	if(num_spe_threads > MAX_SPE_THREADS)
		num_spe_threads = MAX_SPE_THREADS;

	//Find the number of rows each SPE should process
	int rowsPerSPE_Y = cm->mb_rows / num_spe_threads;
	int rowsPerSPE_UV = (cm->mb_rows / 2) / num_spe_threads;

	int start_y_Y = 0;
	int start_y_UV = 0;
	int end_y_Y = rowsPerSPE_Y;
	int end_y_UV = rowsPerSPE_UV;

	//Loop through all the spe threads; divide the data between them and start running them
	for(i = 0; i < num_spe_threads; ++i)
	{
		//Initialize context run data
		ppe_me_data[i].ea_mbY_out = (unsigned long) cm->curframe->mbs[0];
		ppe_me_data[i].ea_mbU_out = (unsigned long) cm->curframe->mbs[1];
		ppe_me_data[i].ea_mbV_out = (unsigned long) cm->curframe->mbs[2];
		ppe_me_data[i].ea_origY_data_in = (unsigned long) cm->curframe->newOrig->Y;
		ppe_me_data[i].ea_origU_data_in = (unsigned long) cm->curframe->newOrig->U;
		ppe_me_data[i].ea_origV_data_in = (unsigned long) cm->curframe->newOrig->V;
		ppe_me_data[i].mb_x = 0;
		ppe_me_data[i].mb_y_Y = start_y_Y;
		ppe_me_data[i].mb_y_UV = start_y_UV;
		ppe_me_data[i].mb_max_y_Y = end_y_Y;
		ppe_me_data[i].mb_max_y_UV = end_y_UV;
		ppe_me_data[i].mb_cols = cm->mb_cols;
		ppe_me_data[i].mb_rows = cm->mb_rows;
		ppe_me_data[i].range = vec_splats((float)cm->me_search_range);
		ppe_me_data[i].padYw = cm->padw[0];
		ppe_me_data[i].padYh = cm->padh[0];
		ppe_me_data[i].padUw = cm->padw[1];
		ppe_me_data[i].padUh = cm->padh[1];
		ppe_me_data[i].padVw = cm->padw[2];
		ppe_me_data[i].padVh = cm->padh[2];
		ppe_me_data[i].ea_reconsY_data_in = (unsigned long) addressesY;
		ppe_me_data[i].ea_reconsU_data_in = (unsigned long) addressesU;
		ppe_me_data[i].ea_reconsV_data_in = (unsigned long) addressesV;

		data[i].argp = (void*) &ppe_me_data[i];

		start_y_Y += rowsPerSPE_Y;
		start_y_UV += rowsPerSPE_UV;
		end_y_Y += rowsPerSPE_Y;
		end_y_UV += rowsPerSPE_UV;

		if(!(data[i].prog = spe_image_open("meblockspe")))
		{
			perror ("Failed to open image");
			sleep(1000);
		}
		if((data[i].spe_ctx = spe_context_create(0, NULL)) == NULL)
		{
			perror ("Failed to create context");
			sleep(1000);
		}
		if(spe_program_load(data[i].spe_ctx, data[i].prog))
		{
			perror ("Failed to load program");
			sleep(1000);
		}
		if(pthread_create(&data[i].pthread, NULL, &ppu_pthread_function, &data[i]))
		{
			perror ("Failed to create thread");
			sleep(1000);
		}
	}

	for(i = 0; i < num_spe_threads; ++i)
	{
		if(pthread_join(data[i].pthread, NULL))
		{
			perror ("Failed to join thread\n");
			sleep(1000);
		}
		if(spe_context_destroy(data[i].spe_ctx))
		{
			perror("Failed to destroy context");
			sleep(1000);
		}
		if(spe_image_close(data[i].prog))
		{
			perror("Failed to close image");
			sleep(1000);
		}
	}
}

/* Motion compensation for 8x8 block */
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *predicted, uint8_t *ref, int cc)
{
  struct macroblock *mb = &cm->curframe->mbs[cc][mb_y * cm->padw[cc]/8 + mb_x];

  if (!mb->use_mv) { return; }

  int left = mb_x * 8;
  int top = mb_y * 8;
  int right = left + 8;
  int bottom = top + 8;

  int w = cm->padw[cc];

  /* Copy block from ref mandated by MV */
  int x, y;
  int i = 0;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      predicted[(top*w)+(left*8+i)] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
      ++i;
    }
  }
}

void c63_motion_compensate(struct c63_common *cm)
{
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y,
          cm->refframe->newRecons->Y, 0);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U,
          cm->refframe->newRecons->U, 1);
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V,
          cm->refframe->newRecons->V, 2);
    }
  }
}
