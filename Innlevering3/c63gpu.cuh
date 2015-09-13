#ifndef C63GPU
#define C63GPU

#include"c63.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

static float final_time = 0;
static cudaEvent_t start_event, end_event;
static cudaStream_t streams[3];

static cudaStream_t quantdeq_streamY[4];
static cudaStream_t quantdeq_streamU[4];
static cudaStream_t quantdeq_streamV[4];

struct me_data
{
	short x;
	short y;
	unsigned short sad;

	__device__ bool operator < (const me_data& other)
	{
		return this->sad < other.sad;
	}
};

struct KernelConfig
{
	dim3 grid_size;
	dim3 block_size;
	cudaStream_t stream;
	cudaStream_t quantdeq_stream[4];
	int size_chunk;
};

struct KernelConfigurations
{
	struct KernelConfig motion_est_Y;
	struct KernelConfig motion_est_U;
	struct KernelConfig motion_est_V;
	struct KernelConfig motion_comp_Y;
	struct KernelConfig motion_comp_U;
	struct KernelConfig motion_comp_V;
	struct KernelConfig quantdequant_Y;
	struct KernelConfig quantdequant_U;
	struct KernelConfig quantdequant_V;

	int n_chunks;
};

struct c63_common_gpu
{
  int width, height;
  int ypw, yph, upw, uph, vpw, vph;

  int padw[3], padh[3];

  int mb_cols, mb_rows;

  uint8_t qp;                         // Quality parameter

  int me_search_range;

  uint8_t quanttbl[3][64];

  int framenum;

  int keyframe_interval;
  int frames_since_keyframe;

  struct entropy_ctx e_ctx;
};

#endif