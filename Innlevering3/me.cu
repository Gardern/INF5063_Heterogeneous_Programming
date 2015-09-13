#include <assert.h>
#include <errno.h>
#ifndef WIN32
#include <getopt.h>
#endif
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "c63cpu.cuh"


// Motion estimation for 8x8 block 
static void me_block_8x8_CPU(struct c63_common_cpu *cm, int mb_x, int mb_y,
    uint8_t *orig, uint8_t *ref, int cc)
{
  struct macroblock *mb = &cm->mbs[cc][mb_y * cm->padw[cc]/8 + mb_x];

  int range = cm->me_search_range;

  // Half resolution for chroma channels.
  if (cc > 0) { range /= 2; }

  int left = mb_x * 8 - range;
  int top = mb_y * 8 - range;
  int right = mb_x * 8 + range;
  int bottom = mb_y * 8 + range;

  int w = cm->padw[cc];
  int h = cm->padh[cc];

  // Make sure we are within bounds of reference frame. TODO: Support partial frame bounds. 
  if (left < 0) { left = 0; }
  if (top < 0) { top = 0; }
  if (right > (w - 8)) { right = w - 8; }
  if (bottom > (h - 8)) { bottom = h - 8; }

  int x, y;

  int mx = mb_x * 8;
  int my = mb_y * 8;

  int best_sad = INT_MAX;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      int sad;
      sad_block_8x8(orig + my*w+mx, ref + y*w+x, w, &sad);

      if (sad < best_sad)
      {
        mb->mv_x = x - mx;
        mb->mv_y = y - my;
        best_sad = sad;
      }
    }
  }

  // Here, there should be a threshold on SAD that checks if the motion vector
   //  is cheaper than intraprediction. We always assume MV to be beneficial 

   //printf("Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y,
   //  best_sad); 

  mb->use_mv = 1;
}

void c63_motion_estimate(struct c63_common_cpu *cm)
{
  // Compare this frame with previous reconstructed frame
  int mb_x, mb_y;

  //Luma
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      me_block_8x8_CPU(cm, mb_x, mb_y, cm->curframe->orig->Y,
          cm->refframe->recons->Y, 0);
    }
  }

  //Chroma 
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      me_block_8x8_CPU(cm, mb_x, mb_y, cm->curframe->orig->U,
          cm->refframe->recons->U, 1);
      me_block_8x8_CPU(cm, mb_x, mb_y, cm->curframe->orig->V,
          cm->refframe->recons->V, 2);
    }
  }
}

void c63_motion_estimate_Y_CPU(struct c63_common_cpu *cm, int cpu_start)
{
	// Compare this frame with previous reconstructed frame
	int mb_x, mb_y;

	//Luma
	#pragma omp parallel for private(mb_x)
	for (mb_y = cpu_start; mb_y < cm->mb_rows; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
		{
			me_block_8x8_CPU(cm, mb_x, mb_y, cm->orig->Y, cm->ref_recons->Y, 0);
		}
	}
}

void c63_motion_estimate_U_CPU(struct c63_common_cpu *cm, int cpu_start)
{
	// Compare this frame with previous reconstructed frame
	int mb_x, mb_y;

	//Luma
	#pragma omp parallel for private(mb_x)
	for (mb_y = cpu_start; mb_y < cm->mb_rows / 2; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
		{
			me_block_8x8_CPU(cm, mb_x, mb_y, cm->orig->U, cm->ref_recons->U, 1);
		}
	}
}

void c63_motion_estimate_V_CPU(struct c63_common_cpu *cm, int cpu_start)
{
	// Compare this frame with previous reconstructed frame
	int mb_x, mb_y;

	//Luma
	#pragma omp parallel for private(mb_x)
	for (mb_y = cpu_start; mb_y < cm->mb_rows / 2; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
		{
			me_block_8x8_CPU(cm, mb_x, mb_y, cm->orig->V, cm->ref_recons->V, 2);
		}
	}
}

// Motion compensation for 8x8 block 
static void mc_block_8x8(struct c63_common_cpu *cm, int mb_x, int mb_y,
    uint8_t *predicted, uint8_t *ref, int cc)
{
  struct macroblock *mb = &cm->curframe->mbs[cc][mb_y * cm->padw[cc]/8 + mb_x];

  if (!mb->use_mv) { return; }

  int left = mb_x * 8;
  int top = mb_y * 8;
  int right = left + 8;
  int bottom = top + 8;

  int w = cm->padw[cc];

  // Copy block from ref mandated by MV 
  int x, y;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      predicted[y*w+x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
    }
  }
}

void c63_motion_compensate(struct c63_common_cpu *cm)
{
  int mb_x, mb_y;

  // Luma *
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y,
          cm->refframe->recons->Y, 0);
    }
  }

   //Chroma 
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U,
          cm->refframe->recons->U, 1);
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V,
          cm->refframe->recons->V, 2);
    }
  }
}