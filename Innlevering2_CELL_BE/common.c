#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "c63.h"
#include "tables.h"

#include <pthread.h>
#include <libspe2.h>

#include "c63.h"

float *testResiduals;

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
	unsigned long long in_data;
	unsigned long long prediction;
	int width;
	int height;
	int y;
	int range;
	unsigned long long out_data;
	unsigned long long quantization;
} dct_parm_data;

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

dct_parm_data ppe_dct_data[MAX_SPE_THREADS] __attribute__ ((aligned (16)));
idct_parm_data ppe_idct_data[MAX_SPE_THREADS] __attribute__ ((aligned (16)));

ppe_pthread_data_t data[MAX_SPE_THREADS];
ppe_pthread_data_t data2[MAX_SPE_THREADS];

void *ppu_pthread_function2(void *arg)
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

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, uint8_t *out_data, uint8_t *quantization)
{
	int i;

	//Get the number of SPEs
	int num_spe_threads = spe_cpu_info_get(SPE_COUNT_USABLE_SPES, -1);
	if(num_spe_threads > MAX_SPE_THREADS)
		num_spe_threads = MAX_SPE_THREADS;

	//Find the number of rows each SPE should process
	int rowsPerSPE = ((height / 8) / num_spe_threads);

	int start = 0;
	int end = rowsPerSPE;

	//Loop through all the spe threads; divide the data between them and start running them
	for(i = 0; i < num_spe_threads; ++i)
	{
		ppe_idct_data[i].in_data = (unsigned long) in_data;
		ppe_idct_data[i].prediction = (unsigned long) prediction;
		ppe_idct_data[i].width = width;
		ppe_idct_data[i].height = height;
		ppe_idct_data[i].y = start;
		ppe_idct_data[i].range = end;
		ppe_idct_data[i].out_data = (unsigned long) out_data;
		ppe_idct_data[i].quantization = (unsigned long) quantization;

		// Initialize context run data
		data2[i].argp = &ppe_idct_data[i];

		start += rowsPerSPE;
		end += rowsPerSPE;

		if(!(data2[i].prog = spe_image_open("dequantizeidctspe")))
		{
			perror ("Failed to open image");
			sleep(1000);
		}
		if((data2[i].spe_ctx = spe_context_create(0, NULL)) == NULL)
		{
			perror ("Failed to create context");
			sleep(1000);
		}
		if(spe_program_load(data2[i].spe_ctx, data2[i].prog))
		{
			perror ("Failed to load program");
			sleep(1000);
		}
		if(pthread_create(&data2[i].pthread, NULL, &ppu_pthread_function2, &data2[i]))
		{
			perror ("Failed to create thread");
			sleep(1000);
		}
	}
	for(i = 0; i < num_spe_threads; ++i)
	{
		if(pthread_join(data2[i].pthread, NULL))
		{
			perror ("Failed to join thread\n");
			sleep(1000);
		}
		if(spe_context_destroy(data2[i].spe_ctx))
		{
			perror("Failed to destroy context");
			sleep(1000);
		}
		if(spe_image_close(data2[i].prog))
		{
			perror("Failed to close image");
			sleep(1000);
		}
	}
}

void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, int16_t *out_data, uint8_t *quantization)
{
	int i;

	//Get the number of SPEs
	int num_spe_threads = spe_cpu_info_get(SPE_COUNT_USABLE_SPES, -1);
	if(num_spe_threads > MAX_SPE_THREADS)
		num_spe_threads = MAX_SPE_THREADS;

	//Find the number of rows each SPE should process
	int rowsPerSPE = (height / 8) / num_spe_threads;

	int start = 0;
	int end = rowsPerSPE;

	//Loop through all the spe threads; divide the data between them and start running them
	for(i = 0; i < num_spe_threads; ++i)
	{
		ppe_dct_data[i].in_data = (unsigned long) in_data;
		ppe_dct_data[i].prediction = (unsigned long) prediction;
		ppe_dct_data[i].width = width;
		ppe_dct_data[i].height = height;
		ppe_dct_data[i].y = start;
		ppe_dct_data[i].range = end;
		ppe_dct_data[i].out_data = (unsigned long) out_data;
		ppe_dct_data[i].quantization = (unsigned long) quantization;

		// Initialize context run data
		data[i].argp = &ppe_dct_data[i];

		start += rowsPerSPE;
		end += rowsPerSPE;

		if(!(data[i].prog = spe_image_open("dctquantizespe")))
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
		if(pthread_create(&data[i].pthread, NULL, &ppu_pthread_function2, &data[i]))
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

void destroy_frame(struct frame *f)
{
  /* First frame doesn't have a reconstructed frame to destroy */
  if (!f) { return; }

  free(f->recons->Y);
  free(f->recons->U);
  free(f->recons->V);
  free(f->recons);

  free(f->newRecons->Y);
  free(f->newRecons->U);
  free(f->newRecons->V);
  free(f->newRecons);

  free(f->residuals->Ydct);
  free(f->residuals->Udct);
  free(f->residuals->Vdct);
  free(f->residuals);

  free(f->predicted->Y);
  free(f->predicted->U);
  free(f->predicted->V);
  free(f->predicted);

  free(f->mbs[0]);
  free(f->mbs[1]);
  free(f->mbs[2]);

  free(f);
}

struct frame* create_frame(struct c63_common *cm, yuv_t *image, yuv_t *newImage)
{
  struct frame *f = malloc(sizeof(struct frame));

  f->orig = image;
  f->newOrig = newImage;

  f->recons = malloc(sizeof(yuv_t));
  f->recons->Y = malloc(cm->ypw * cm->yph);
  f->recons->U = malloc(cm->upw * cm->uph);
  f->recons->V = malloc(cm->vpw * cm->vph);

  f->newRecons = malloc(sizeof(yuv_t));
  f->newRecons->Y = malloc(cm->ypw * cm->yph);
  f->newRecons->U = malloc(cm->upw * cm->uph);
  f->newRecons->V = malloc(cm->vpw * cm->vph);

  f->predicted = memalign(16, sizeof(yuv_t));
  f->predicted->Y = memalign(16, cm->ypw * cm->yph *sizeof(uint8_t));
  f->predicted->U = memalign(16, cm->upw * cm->uph *sizeof(uint8_t));
  f->predicted->V = memalign(16, cm->vpw * cm->vph * sizeof(uint8_t));

  f->residuals = memalign(16, sizeof(dct_t));
  f->residuals->Ydct = (int16_t*) memalign(16, cm->ypw * cm->yph * sizeof(int16_t));
  f->residuals->Udct = memalign(16, cm->upw * cm->uph * sizeof(int16_t));
  f->residuals->Vdct = memalign(16, cm->vpw * cm->vph * sizeof(int16_t));

  f->mbs[0] = calloc(cm->mb_rows * cm->mb_cols, sizeof(struct macroblock));
  f->mbs[1] = calloc(cm->mb_rows/2 * cm->mb_cols/2, sizeof(struct macroblock));
  f->mbs[2] = calloc(cm->mb_rows/2 * cm->mb_cols/2, sizeof(struct macroblock));

  return f;
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
  fwrite(image->Y, 1, w*h, fp);
  fwrite(image->U, 1, w*h/4, fp);
  fwrite(image->V, 1, w*h/4, fp);
}
