#include <assert.h>
#include <errno.h>
#ifndef WIN32
#include <getopt.h>
#else
#include<iostream>
#include<cstdio>
#include<Windows.h>
#include<boost\math\special_functions\round.hpp>
using namespace std;
#endif
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <math.h>

#include "c63gpu.cuh"
#include "c63cpu.cuh"
#include "tables.cuh"

static char *output_file, *input_file;
FILE *outfile;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;

/* getopt */
#ifndef WIN32
extern int optind;
extern char *optarg;
#endif

//Set this to 1 to enable heterogeneous processing where both the GPU and CPU process its own part of the domain
//(Works best for older GPUs in combination with a good CPU)
//Set it to 0 to let the GPU process everything
//(Works best for the most powerful GPUs that outruns the CPU)
#define USE_HETEROGENEOUS_PROCESSING 1

__device__ uint8_t zigzag_U_GPU[64] =
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

__device__ uint8_t zigzag_V_GPU[64] =
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

__device__ float dctlookup_GPU_dct[8][8] =
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

__constant__ float dctlookup_GPU_idct[8][8] =
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

__device__ uint8_t quanttbl1D[2][64] =
{
	// Y
	{ 16/2.5, 11/2.5, 12/2.5, 14/2.5, 12/2.5, 10/2.5, 16/2.5, 14/2.5,
      13/2.5, 14/2.5, 18/2.5, 17/2.5, 16/2.5, 19/2.5, 24/2.5, 40/2.5,
	  26/2.5, 24/2.5, 22/2.5, 22/2.5, 24/2.5, 49/2.5, 35/2.5, 37/2.5,
	  29/2.5, 40/2.5, 58/2.5, 51/2.5, 61/2.5, 30/2.5, 57/2.5, 51/2.5,
	  56/2.5, 55/2.5, 64/2.5, 72/2.5, 92/2.5, 78/2.5, 64/2.5, 68/2.5,
	  87/2.5, 69/2.5, 55/2.5, 56/2.5, 80/2.5, 109/2.5, 81/2.5, 87/2.5,
	  95/2.5, 98/2.5, 103/2.5, 104/2.5, 103/2.5, 62/2.5, 77/2.5, 113/2.5,
	  121/2.5, 112/2.5, 100/2.5, 120/2.5, 92/2.5, 101/2.5, 103/2.5, 99/2.5,
	},
	// U, V
	{ 17/2.5, 18/2.5, 18/2.5, 24/2.5, 21/2.5, 24/2.5, 47/2.5, 26/2.5,
	  26/2.5, 47/2.5, 99/2.5, 66/2.5, 56/2.5, 66/2.5, 99/2.5, 99/2.5,
	  99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5,
	  99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5,
	  99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5,
	  99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5,
	  99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5,
	  99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5, 99/2.5,
	},
};

__constant__ float ISQRT2 = 0.70710678118654f;
__constant__ float SCALE_VALUE = 1.0f;
__constant__ float QUANT_VALUE = 4.0f;

//Inits the grid and block sizes of all kernels
struct KernelConfigurations init_kernel_configurations(uint32_t width, uint32_t height, struct c63_common_cpu *cm)
{
	struct KernelConfigurations kernel_configs;

	// Set number and size of chunks for quantize and dequantize
	kernel_configs.n_chunks = 1;
	kernel_configs.quantdequant_Y.size_chunk = (cm->ypw * cm->yph) / kernel_configs.n_chunks;
	kernel_configs.quantdequant_U.size_chunk = (cm->upw * cm->uph) / kernel_configs.n_chunks;
	kernel_configs.quantdequant_V.size_chunk = (cm->vpw * cm->vph) / kernel_configs.n_chunks;

	//Set grid and block sizes for all the kernels
	kernel_configs.quantdequant_Y.grid_size = dim3(cm->ypw / 8, (cm->yph / 8) / kernel_configs.n_chunks, 1);
	kernel_configs.quantdequant_Y.block_size = dim3(8, 8, 1);
	kernel_configs.quantdequant_U.grid_size = dim3(cm->upw / 8, (cm->uph / 8) / kernel_configs.n_chunks, 1);
	kernel_configs.quantdequant_U.block_size = dim3(8, 8, 1);
	kernel_configs.quantdequant_V.grid_size = dim3(cm->vpw / 8, (cm->vph / 8) / kernel_configs.n_chunks, 1);
	kernel_configs.quantdequant_V.block_size = dim3(8, 8, 1);
	
	if(width == 352)
	{
		kernel_configs.motion_est_Y.grid_size = dim3(64, 48);
		kernel_configs.motion_est_Y.block_size = dim3(32, 32);
		kernel_configs.motion_est_U.grid_size = dim3(64 / 2, 48 / 2);
		kernel_configs.motion_est_U.block_size = dim3(16, 16);
		kernel_configs.motion_est_V.grid_size = dim3(64 / 2, 48 / 2);
		kernel_configs.motion_est_V.block_size = dim3(16, 16);

		kernel_configs.motion_comp_Y.grid_size = dim3(64, 48);
		kernel_configs.motion_comp_Y.block_size = dim3(8, 8);
		kernel_configs.motion_comp_U.grid_size = dim3(64 / 2, 48 / 2);
		kernel_configs.motion_comp_U.block_size = dim3(8, 8);
		kernel_configs.motion_comp_V.grid_size = dim3(64 / 2, 48 / 2);
		kernel_configs.motion_comp_V.block_size = dim3(8, 8);
	}
	else if(width == 1920)
	{
		kernel_configs.motion_est_Y.grid_size = dim3(256, 144);
		kernel_configs.motion_est_Y.block_size = dim3(32, 32);
		kernel_configs.motion_est_U.grid_size = dim3(256 / 2, 144 / 2);
		kernel_configs.motion_est_U.block_size = dim3(16, 16);
		kernel_configs.motion_est_V.grid_size = dim3(256 / 2, 144 / 2);
		kernel_configs.motion_est_V.block_size = dim3(16, 16);

		kernel_configs.motion_comp_Y.grid_size = dim3(256, 144);
		kernel_configs.motion_comp_Y.block_size = dim3(8, 8);
		kernel_configs.motion_comp_U.grid_size = dim3(256 / 2, 144 / 2);
		kernel_configs.motion_comp_U.block_size = dim3(8, 8);
		kernel_configs.motion_comp_V.grid_size = dim3(256 / 2, 144 / 2);
		kernel_configs.motion_comp_V.block_size = dim3(8, 8);
	}

	//Allocate and set streams for all kernels
	for(int i = 0; i < 3; ++i)
	{
		cudaStreamCreate(&streams[i]);
	}

	for(int i = 0; i < kernel_configs.n_chunks; ++i)
	{
		cudaStreamCreate(&quantdeq_streamY[i]);
		cudaStreamCreate(&quantdeq_streamU[i]);
		cudaStreamCreate(&quantdeq_streamV[i]);

		kernel_configs.quantdequant_Y.quantdeq_stream[i] = streams[0];
		kernel_configs.quantdequant_U.quantdeq_stream[i] = streams[1];
		kernel_configs.quantdequant_V.quantdeq_stream[i] = streams[2];
	}


	kernel_configs.motion_est_Y.stream = streams[0];
	kernel_configs.motion_est_U.stream = streams[1];
	kernel_configs.motion_est_V.stream = streams[2];

	kernel_configs.motion_comp_Y.stream = streams[0];
	kernel_configs.motion_comp_U.stream = streams[1];
	kernel_configs.motion_comp_V.stream = streams[2];

	kernel_configs.quantdequant_Y.stream = streams[0];
	kernel_configs.quantdequant_U.stream = streams[1];
	kernel_configs.quantdequant_V.stream = streams[2];

	return kernel_configs;
}

//Partition the domain between the CPU and GPU
void partition_domain(uint32_t width, uint32_t height, struct c63_common_cpu *cm)
{
	//Hardcode the best partition for foreman and tractor
	if(width == 352)
	{
		split_atY = 12;
		gpu_partY = 12;
		cpu_partY = cm->mb_rows - gpu_partY;

		split_atUV = 4;
		gpu_partUV = 4;
		cpu_partUV = cm->mb_rows / 2 - gpu_partUV;
	}
	else if(width == 1920)
	{
		split_atY = 40;
		gpu_partY = 40;
		cpu_partY = cm->mb_rows - gpu_partY;

		split_atUV = 22;
		gpu_partUV = 22;
		cpu_partUV = cm->mb_rows / 2 - gpu_partUV;
	}
}

__device__ struct me_data min_gpu(struct me_data a, struct me_data b)
{
	return ((a) < (b) ? (a) : (b));
}

__device__  bool inside_domain(int x, int y, int w, int h)
{
	if((x < 0 || y < 0) || (x >= (w - 8) || y >= (h - 8)))
		return false;

	return true;
}

void print_GPU_info()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	
	printf("Device name: %s\n", prop.name);
	printf("Global memory (bytes): %d\n", prop.totalGlobalMem);
	printf("Constant memory (bytes): %d\n", prop.totalConstMem);
	printf("Shared memory per block (bytes): %d\n", prop.sharedMemPerBlock);
	printf("Compute mode: %d\n", prop.computeMode);

	if(prop.concurrentKernels == 1)
		printf("%s\n", "Can execute kernels concurrently");
	else
		printf("%s\n", "Can not execute kernels concurrently");

	printf("%s\n", "\n");
}

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static yuv_t* read_yuv(FILE *file, struct c63_common_cpu *cm)
{
  size_t len = 0;
  yuv_t *image = (yuv_t*) malloc(sizeof(*image));

  cudaMallocHost((void**)&image->Y, cm->padw[0]*cm->padh[0]);
  cudaMallocHost((void**)&image->U, cm->padw[1]*cm->padh[1]);
  cudaMallocHost((void**)&image->V, cm->padw[2]*cm->padh[2]);

  /* Read Y. The size of Y is the same as the size of the image. The indices
     represents the color component (0 is Y, 1 is U, and 2 is V) */
  len += fread(image->Y, 1, width*height, file);

  /* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y
     because (height/2)*(width/2) = (height*width)/4. */
  len += fread(image->U, 1, (width*height)/4, file);

  /* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
  len += fread(image->V, 1, (width*height)/4, file);

  if (ferror(file))
  {
    perror("ferror");
    exit(EXIT_FAILURE);
  }

  if (feof(file))
  {
	cudaFreeHost((void*)image->Y);
    cudaFreeHost((void*)image->U);
	cudaFreeHost((void*)image->V);
    free(image);

    return NULL;
  }
  else if (len != width*height*1.5)
  {
    fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
    fprintf(stderr, "Wrong input? (height: %d width: %d)\n", height, width);

	cudaFreeHost((void*)image->Y);
    cudaFreeHost((void*)image->U);
	cudaFreeHost((void*)image->V);
    free(image);

    return NULL;
  }

  return image;
}

template<int origX, int origY, int refX, int refY> __device__ void sad_block_8x8_GPU(uint8_t (&b1)[origY][origX], uint8_t (&b2)[refY][refX], int stride, unsigned short *result, int tid_x, int tid_y) //int tid_x, int tid_y
{
	int u1 = 0;
	int v1 = 0;
	int u2 = tid_x;
	int v2 = tid_y;

	*result = 0;

	for (v1 = 0, v2 = tid_y; v1 < 8; ++v1, ++v2)
	{
		for (u1 = 0, u2 = tid_x; u1 < 8; ++u1, ++u2)
		{
			*result = (unsigned short)__usad(b2[v2][u2], b1[v1][u1], *result);
		}
	}
}

template<int threads, int origX, int origY, int refX, int refY> __device__ void me_block_8x8_GPU(const uint8_t *orig, const uint8_t  *ref, struct macroblock *mbs, int tid_x, int tid_y, int width, int height, int bid_x, int bid_y, 
								short x, short y, int horizontal_range, const unsigned int border)
{
	__shared__ struct me_data data_shared[threads];
	__shared__ uint8_t b1[origY][origX];
	__shared__ uint8_t b2[refY][refX];

	struct macroblock *mb = &mbs[bid_y * width / 8 + bid_x];
	int tid = tid_y * horizontal_range + tid_x;

	short mx = bid_x * 8;
	short my = bid_y * 8;

	unsigned short sad = INT_MAX;

	//Put orig in shared memory
	if(tid_x < 8 && tid_y < 8)
	{
		b1[tid_y][tid_x] = (orig + (tid_y+my)*width+(tid_x+mx))[0];
	}
	
	//Use all threads to put ref in shared memory
	b2[tid_y][tid_x] = (ref + y*width+x)[0];
	
	//Last eight columns of ref
	if(tid_x < 7)
		b2[tid_y][tid_x+border] = (ref + y*width+(x+border))[0];

	//Last eight rows of ref
	if(tid_y < 7)
		b2[tid_y+border][tid_x] = (ref + (y+border)*width+x)[0];
	
	//Bottom right block of ref
	if(tid_x < 7 && tid_y < 7)
		b2[tid_y+border][tid_x+border] = (ref + (y+border)*width+(x+border))[0];
	
	__syncthreads();
	
	if(inside_domain(x, y, width, height))
	{
		sad_block_8x8_GPU<origX, origY, refX, refY>(b1, b2, width, &sad, tid_x, tid_y);
	}

	data_shared[tid].sad = sad;
	data_shared[tid].x = x;
	data_shared[tid].y = y;
	__syncthreads();
	
	//Use all threads to do a reduction to find the minimum sad
	if(threads >= 1024)
	{
		if(tid < 512) { data_shared[tid] = min_gpu(data_shared[tid], data_shared[tid + 512]); } __syncthreads();
	}
	if(threads >= 512)
	{
		if(tid < 256) { data_shared[tid] = min_gpu(data_shared[tid], data_shared[tid + 256]); } __syncthreads();
	}
	if(threads >= 256)
	{
		if(tid < 128) { data_shared[tid] = min_gpu(data_shared[tid], data_shared[tid + 128]); } __syncthreads();
	}
	if(threads >= 128)
	{
		if(tid < 64) { data_shared[tid] = min_gpu(data_shared[tid], data_shared[tid + 64]); } __syncthreads();
	}
	if(threads >= 64)
	{
		if(tid < 32) { data_shared[tid] = min_gpu(data_shared[tid], data_shared[tid + 32]); } __syncthreads();
	}
	if(threads >= 32)
	{
		if(tid < 16) { data_shared[tid] = min_gpu(data_shared[tid], data_shared[tid + 16]); } __syncthreads();
	}
	if(threads >= 16)
	{
		if(tid < 8) { data_shared[tid] = min_gpu(data_shared[tid], data_shared[tid + 8]); } __syncthreads();
	}
	if(threads >= 8)
	{
		if(tid < 4) { data_shared[tid] = min_gpu(data_shared[tid], data_shared[tid + 4]); } __syncthreads();
	}
	if(threads >= 4)
	{
		if(tid < 2) { data_shared[tid] = min_gpu(data_shared[tid], data_shared[tid + 2]); } __syncthreads();
	}
	if(threads >= 2)
	{
		if(tid < 1) { data_shared[tid] = min_gpu(data_shared[tid], data_shared[tid + 1]); } __syncthreads();
	}
	
	if(tid == 0)
	{
		mb->mv_x = data_shared[tid].x - mx;
		mb->mv_y = data_shared[tid].y - my;
		mb->use_mv = 1;
	}
}

__device__ void mc_block_8x8_GPU(uint8_t *predicted, const uint8_t *ref, const struct macroblock *mbs, int cc, int x, int y, int width, unsigned int bid_x, unsigned int bid_y)
{
	int mb_number = bid_y * width / 8 + bid_x;
	struct macroblock mb = mbs[mb_number];

	if (!mb.use_mv) { return; }
	
	predicted[y*width+x] = ref[(y + mb.mv_y) * width + (x + mb.mv_x)];
}

__global__ void c63_motion_estimate_Y_GPU(uint8_t* origY, uint8_t* reconsY, struct macroblock *mbsY, int width, int height, int cols, int rows, int search_range)
{
	if(blockIdx.x < cols && blockIdx.y < rows)
	{
		int range = search_range;

		int left = blockIdx.x * 8 - range;
		int top = blockIdx.y * 8 - range;
		int right = blockIdx.x * 8 + range;
		int bottom = blockIdx.y * 8 + range;

		int w = width;
		int h = height;

		short x = threadIdx.x + left;
		short y = threadIdx.y + top;
		int horizontal_range = right - left;
		int vertical_range = bottom - top;

		me_block_8x8_GPU<1024, 8, 8, 39, 39>(origY, reconsY, mbsY, threadIdx.x, threadIdx.y, w, h, blockIdx.x, blockIdx.y, x, y, horizontal_range, 32);
	}
}

__global__ void c63_motion_estimate_U_GPU(uint8_t* origU, uint8_t* reconsU, struct macroblock *mbsU, int width, int height, int cols, int rows, int search_range)
{
	if(blockIdx.x < cols && blockIdx.y < rows)
	{
		int range = search_range;
		range /= 2;

		int left = blockIdx.x * 8 - range;
		int top = blockIdx.y * 8 - range;
		int right = blockIdx.x * 8 + range;
		int bottom = blockIdx.y * 8 + range;

		int w = width;
		int h = height;

		short x = threadIdx.x + left;
		short y = threadIdx.y + top;
		int horizontal_range = right - left;
		int vertical_range = bottom - top;

		me_block_8x8_GPU<256, 8, 8, 23, 23>(origU, reconsU, mbsU, threadIdx.x, threadIdx.y, w, h, blockIdx.x, blockIdx.y, x, y, horizontal_range, 16);
	}
}

__global__ void c63_motion_estimate_V_GPU(uint8_t* origV, uint8_t* reconsV, struct macroblock *mbsV, int width, int height, int cols, int rows, int search_range)
{
	if(blockIdx.x < cols && blockIdx.y < rows)
	{
		int range = search_range;
		range /= 2;

		int left = blockIdx.x * 8 - range;
		int top = blockIdx.y * 8 - range;
		int right = blockIdx.x * 8 + range;
		int bottom = blockIdx.y * 8 + range;

		int w = width;
		int h = height;

		short x = threadIdx.x + left;
		short y = threadIdx.y + top;
		int horizontal_range = right - left;
		int vertical_range = bottom - top;

		me_block_8x8_GPU<256, 8, 8, 23, 23>(origV, reconsV, mbsV, threadIdx.x, threadIdx.y, w, h, blockIdx.x, blockIdx.y, x, y, horizontal_range, 16);
	}
}

__global__ void c63_motion_compensate_Y_GPU(uint8_t* predictedY, uint8_t* reconsY, struct macroblock *mbsY, int width, int cols, int rows)
{
	int idx, idy;
	
	if(blockIdx.x < cols && blockIdx.y < rows)
	{
		idx = blockIdx.x * 8 + threadIdx.x;
		idy = blockIdx.y * 8 + threadIdx.y;

		mc_block_8x8_GPU(predictedY, reconsY, mbsY, 0, idx, idy, width, blockIdx.x, blockIdx.y);
	}
}

__global__ void c63_motion_compensate_U_GPU(uint8_t* predictedU, uint8_t* reconsU, struct macroblock *mbsU, int width, int cols, int rows)
{
	int idx, idy;

	if(blockIdx.x < cols && blockIdx.y < rows)
	{
		idx = blockIdx.x * 8 + threadIdx.x;
		idy = blockIdx.y * 8 + threadIdx.y;

		mc_block_8x8_GPU(predictedU, reconsU, mbsU, 1, idx, idy, width, blockIdx.x, blockIdx.y);
	}
}

__global__ void c63_motion_compensate_V_GPU(uint8_t* predictedV, uint8_t* reconsV, struct macroblock *mbsV, int width, int cols, int rows)
{
	int idx, idy;

	if(blockIdx.x < cols && blockIdx.y < rows)
	{
		idx = blockIdx.x * 8 + threadIdx.x;
		idy = blockIdx.y * 8 + threadIdx.y;

		mc_block_8x8_GPU(predictedV, reconsV, mbsV, 2, idx, idy, width, blockIdx.x, blockIdx.y);
	}
}

__device__ static void transpose_block_GPU(float *in_data, float *out_data)
{
	int i = threadIdx.y;
	int j = threadIdx.x;
	
	out_data[i*8+j] = in_data[j*8+i];
}

__shared__ float a1;
__shared__ float a2;

__device__ static void scale_block_GPU(float *in_data, float *out_data)
{
	int v = threadIdx.y;
	int u = threadIdx.x;

	a1 = !u ? ISQRT2 : SCALE_VALUE;
	a2 = !v ? ISQRT2 : SCALE_VALUE;

	// Scale according to normalizing function 
	out_data[v*8+u] = in_data[v*8+u] * a1 * a2;
}

__shared__ float dct;

__device__ static void dct_1d_GPU(float *in_data, float *out_data)
{
	int j;
	dct = 0;

    for (j = 0; j < 8; ++j)
    {
		dct += in_data[j] * dctlookup_GPU_dct[j][threadIdx.x];
    }

    out_data[threadIdx.x] = dct;
}

__device__ static void quantize_block_GPU(float *in_data, float *out_data, int index)
{
	int tid = threadIdx.y * 8 + threadIdx.x;

    uint8_t u = zigzag_U_GPU[tid];
    uint8_t v = zigzag_V_GPU[tid];

	dct = in_data[v*8+u];

    // Zig-zag and quantize 
	out_data[tid] = (float) roundf((dct /  QUANT_VALUE) / quanttbl1D[index][tid]);
}
__shared__ float mb[8*8];
__shared__ float mb2[8*8];

__device__ void dct_quant_block_8x8_GPU(uint8_t *in_data, uint8_t *prediction, int w, int16_t *out_data, int index)
{
  	int i = threadIdx.y;
	int j = threadIdx.x;
	int tid = i * 8 + j;

    mb2[i*8+j] = ((float)in_data[i*w+j] - prediction[i*w+j]);
    
  // Two 1D DCT operations with transpose 
	dct_1d_GPU(mb2+i*8, mb+i*8);
	__syncthreads();
	transpose_block_GPU(mb, mb2);
	
	__syncthreads();

	dct_1d_GPU(mb2+i*8, mb+i*8);
	__syncthreads();
	transpose_block_GPU(mb, mb2);

	__syncthreads();

	scale_block_GPU(mb2, mb);

	__syncthreads();

	quantize_block_GPU(mb, mb2, index);

	out_data[tid] = mb2[tid];
}

__global__ void dct_quantize_GPU(uint8_t *in_data, uint8_t *prediction, uint32_t width, int16_t *out_data, int index)
{
	int bid = blockIdx.y * (width / 8) + blockIdx.x;
	int x = blockIdx.x * 8;
	int y = blockIdx.y * width * 8;
	
	dct_quant_block_8x8_GPU(in_data+(y+x), prediction+(y+x), width, out_data+(bid*64), index);
}

__shared__ float idct;

__device__ static void idct_1d_GPU(float *in_data, float *out_data)
{
	int j;
    idct = 0;

    for (j = 0; j < 8; ++j)
    {
		idct += in_data[j] * dctlookup_GPU_idct[threadIdx.y][j];
    }

	out_data[threadIdx.y] = idct;
}

__device__ static void dequantize_block_GPU(float *in_data, float *out_data, int index)
{
	int tid = threadIdx.y * 8 + threadIdx.x;

    uint8_t u = zigzag_U_GPU[tid];
    uint8_t v = zigzag_V_GPU[tid];

	dct = in_data[tid];

    // Zig-zag and de-quantize 
	out_data[v*8+u] = (float) roundf((dct * quanttbl1D[index][tid]) / QUANT_VALUE);
}

__shared__ float mbDeq[8*8];
__shared__ float mbDeq2[8*8];
__shared__ int16_t tmp[8*8];

__device__ void dequant_idct_block_8x8_GPU(int16_t *in_data, uint8_t *prediction, int w, uint8_t *out_data, int index)
{
	int i = threadIdx.y;
	int j = threadIdx.x;
	int tid = i * 8 + j;
	
	mbDeq[tid] = in_data[tid];
	
	dequantize_block_GPU(mbDeq, mbDeq2, index);

	__syncthreads();

	scale_block_GPU(mbDeq2, mbDeq);
	
	__syncthreads();

	// Two 1D inverse DCT operations with transpose 
	idct_1d_GPU(mbDeq+j*8, mbDeq2+j*8);
	__syncthreads();
	transpose_block_GPU(mbDeq2, mbDeq);

	__syncthreads();
	
	idct_1d_GPU(mbDeq+j*8, mbDeq2+j*8);
	__syncthreads();
	transpose_block_GPU(mbDeq2, mbDeq);

	// Add prediction block. Note: DCT is not precise - Clamp to legal values
	tmp[i*8+j] = (int16_t)mbDeq[i*8+j] + (int16_t)prediction[i*w+j];
	if (tmp[i*8+j] < 0) { tmp[i*8+j] = 0; }
	else if (tmp[i*8+j] > 255) { tmp[i*8+j] = 255; }
	
	out_data[i*w+j] = tmp[i*8+j];
}

__global__ void dequantize_idct_GPU(int16_t *in_data, uint8_t *prediction, uint32_t width, uint8_t *out_data, int index)
{
	int bid = blockIdx.y * (width / 8) + blockIdx.x;
	int x = blockIdx.x * 8;
	int y = blockIdx.y * width * 8;

	dequant_idct_block_8x8_GPU(in_data+(bid*64), prediction+(y+x), width, out_data+(y+x), index);
}

static void c63_encode_image(struct c63_common_cpu *cm_cpu, struct c63_common_gpu *cm_gpu, struct KernelConfigurations kernel_configs, uint8_t* d_origY, uint8_t* d_origU, uint8_t* d_origV, uint8_t* d_ref_reconsY, uint8_t* d_ref_reconsU, uint8_t* d_ref_reconsV, 
								uint8_t* d_current_reconsY, uint8_t* d_current_reconsU, uint8_t* d_current_reconsV, uint8_t* d_predictedY, uint8_t* d_predictedU, uint8_t* d_predictedV, 
									struct macroblock *d_mbsY, struct macroblock *d_mbsU, struct macroblock *d_mbsV, int16_t *d_residualsYDCT, int16_t *d_residualsUDCT, int16_t *d_residualsVDCT,
										uint8_t *d_quant0, uint8_t *d_quant1, uint8_t *d_quant2)
{
	float me_time = 0;
	double time = 0;

	cudaMemset((void*)d_mbsY, 0, cm_gpu->mb_rows * cm_gpu->mb_cols * sizeof(struct macroblock));
	cudaMemset((void*)d_mbsU, 0, cm_gpu->mb_rows/2 * cm_gpu->mb_cols/2 * sizeof(struct macroblock));
	cudaMemset((void*)d_mbsV, 0, cm_gpu->mb_rows/2 * cm_gpu->mb_cols/2 * sizeof(struct macroblock));

	memset((void*)cm_cpu->mbs[0], 0, cm_gpu->mb_rows * cm_gpu->mb_cols * sizeof(struct macroblock));
	memset((void*)cm_cpu->mbs[1], 0, cm_gpu->mb_rows/2 * cm_gpu->mb_cols/2 * sizeof(struct macroblock));
	memset((void*)cm_cpu->mbs[2], 0, cm_gpu->mb_rows/2 * cm_gpu->mb_cols/2 * sizeof(struct macroblock));

	/* Check if keyframe */
	if (cm_gpu->framenum == 0 || cm_gpu->frames_since_keyframe == cm_gpu->keyframe_interval)
	{
		cm_cpu->keyframe = 1;
		cm_gpu->frames_since_keyframe = 0;

		fprintf(stderr, " (keyframe) ");

		cudaMemcpyAsync(d_origY, cm_cpu->orig->Y, cm_gpu->padw[0]*cm_gpu->padh[0], cudaMemcpyHostToDevice, kernel_configs.motion_est_Y.stream);
		cudaMemcpyAsync(d_origU, cm_cpu->orig->U, cm_gpu->padw[1]*cm_gpu->padh[1], cudaMemcpyHostToDevice, kernel_configs.motion_est_U.stream);
		cudaMemcpyAsync(d_origV, cm_cpu->orig->V, cm_gpu->padw[2]*cm_gpu->padh[2], cudaMemcpyHostToDevice, kernel_configs.motion_est_V.stream);

		cudaMemset((void*)d_predictedY, 0, cm_gpu->ypw * cm_gpu->yph * sizeof(uint8_t));
		cudaMemset((void*)d_predictedU, 0, cm_gpu->upw * cm_gpu->uph * sizeof(uint8_t));
		cudaMemset((void*)d_predictedV, 0, cm_gpu->vpw * cm_gpu->vph * sizeof(uint8_t));
	}
	else { cm_cpu->keyframe = 0; }

	if (!cm_cpu->keyframe)
	{
		d_ref_reconsY = d_current_reconsY;
		d_ref_reconsU = d_current_reconsU;
		d_ref_reconsV = d_current_reconsV;

		cudaMemcpyAsync(cm_cpu->ref_recons->Y, d_current_reconsY, cm_gpu->ypw * cm_gpu->yph, cudaMemcpyDeviceToHost, kernel_configs.motion_est_Y.stream);
		cudaMemcpyAsync(cm_cpu->ref_recons->U, d_current_reconsU, cm_gpu->upw * cm_gpu->uph, cudaMemcpyDeviceToHost, kernel_configs.motion_est_U.stream);
		cudaMemcpyAsync(cm_cpu->ref_recons->V, d_current_reconsV, cm_gpu->vpw * cm_gpu->vph, cudaMemcpyDeviceToHost, kernel_configs.motion_est_V.stream);

		cudaEventRecord(start_event, 0);
#if USE_HETEROGENEOUS_PROCESSING
		cudaMemcpyAsync(d_origY, cm_cpu->orig->Y, cm_gpu->padw[0]*cm_gpu->padh[0], cudaMemcpyHostToDevice, kernel_configs.motion_est_Y.stream); //TRANSFER BOTTOM HALF HERE
		c63_motion_estimate_Y_GPU<<<kernel_configs.motion_est_Y.grid_size, kernel_configs.motion_est_Y.block_size, 0, kernel_configs.motion_est_Y.stream>>>
		(d_origY, d_ref_reconsY, d_mbsY, cm_gpu->width, cm_gpu->height, cm_gpu->mb_cols, split_atY, cm_gpu->me_search_range);

		c63_motion_estimate_Y_CPU(cm_cpu, split_atY);
		cudaMemcpyAsync(&d_mbsY[split_atY*cm_gpu->mb_cols], &cm_cpu->mbs[0][split_atY*cm_gpu->mb_cols], cpu_partY*cm_gpu->mb_cols*sizeof(struct macroblock), cudaMemcpyHostToDevice, kernel_configs.motion_est_Y.stream);

		cudaMemcpyAsync(d_origU, cm_cpu->orig->U, cm_gpu->padw[1]*cm_gpu->padh[1], cudaMemcpyHostToDevice, kernel_configs.motion_est_U.stream);
		c63_motion_estimate_U_GPU<<<kernel_configs.motion_est_U.grid_size, kernel_configs.motion_est_U.block_size, 0, kernel_configs.motion_est_U.stream>>>
		(d_origU, d_ref_reconsU, d_mbsU, cm_gpu->width / 2, cm_gpu->height / 2, cm_gpu->mb_cols / 2, split_atUV, cm_gpu->me_search_range);

		c63_motion_estimate_U_CPU(cm_cpu, split_atUV);
		cudaMemcpyAsync(&d_mbsU[split_atUV*(cm_gpu->mb_cols/2)], &cm_cpu->mbs[1][split_atUV*(cm_gpu->mb_cols/2)], cpu_partUV*cm_gpu->mb_cols/2*sizeof(struct macroblock), cudaMemcpyHostToDevice, kernel_configs.motion_est_U.stream);

		cudaMemcpyAsync(d_origV, cm_cpu->orig->V, cm_gpu->padw[2]*cm_gpu->padh[2], cudaMemcpyHostToDevice, kernel_configs.motion_est_V.stream);
		c63_motion_estimate_V_GPU<<<kernel_configs.motion_est_V.grid_size, kernel_configs.motion_est_V.block_size, 0, kernel_configs.motion_est_V.stream>>>
		(d_origV, d_ref_reconsV, d_mbsV, cm_gpu->width / 2, cm_gpu->height / 2, cm_gpu->mb_cols / 2, split_atUV, cm_gpu->me_search_range);

		c63_motion_estimate_V_CPU(cm_cpu, split_atUV);
		cudaMemcpyAsync(&d_mbsV[split_atUV*(cm_gpu->mb_cols/2)], &cm_cpu->mbs[2][split_atUV*(cm_gpu->mb_cols/2)], cpu_partUV*cm_gpu->mb_cols/2*sizeof(struct macroblock), cudaMemcpyHostToDevice, kernel_configs.motion_est_V.stream);
#else
		cudaMemcpyAsync(d_origY, cm_cpu->orig->Y, cm_gpu->padw[0]*cm_gpu->padh[0], cudaMemcpyHostToDevice, kernel_configs.motion_est_Y.stream); //TRANSFER BOTTOM HALF HERE
		c63_motion_estimate_Y_GPU<<<kernel_configs.motion_est_Y.grid_size, kernel_configs.motion_est_Y.block_size, 0, kernel_configs.motion_est_Y.stream>>>
		(d_origY, d_ref_reconsY, d_mbsY, cm_gpu->width, cm_gpu->height, cm_gpu->mb_cols, cm_gpu->mb_rows, cm_gpu->me_search_range);

		cudaMemcpyAsync(d_origU, cm_cpu->orig->U, cm_gpu->padw[1]*cm_gpu->padh[1], cudaMemcpyHostToDevice, kernel_configs.motion_est_U.stream);
		c63_motion_estimate_U_GPU<<<kernel_configs.motion_est_U.grid_size, kernel_configs.motion_est_U.block_size, 0, kernel_configs.motion_est_U.stream>>>
		(d_origU, d_ref_reconsU, d_mbsU, cm_gpu->width / 2, cm_gpu->height / 2, cm_gpu->mb_cols / 2, cm_gpu->mb_rows / 2, cm_gpu->me_search_range);

		cudaMemcpyAsync(d_origV, cm_cpu->orig->V, cm_gpu->padw[2]*cm_gpu->padh[2], cudaMemcpyHostToDevice, kernel_configs.motion_est_V.stream);
		c63_motion_estimate_V_GPU<<<kernel_configs.motion_est_V.grid_size, kernel_configs.motion_est_V.block_size, 0, kernel_configs.motion_est_V.stream>>>
		(d_origV, d_ref_reconsV, d_mbsV, cm_gpu->width / 2, cm_gpu->height / 2, cm_gpu->mb_cols / 2, cm_gpu->mb_rows / 2, cm_gpu->me_search_range);
#endif

		cudaEventRecord(end_event, 0);
		cudaEventSynchronize(end_event);
		cudaEventElapsedTime(&me_time, start_event, end_event);
		printf("ME kernel time: %f\n", me_time);

#if USE_HETEROGENEOUS_PROCESSING
		//Transfer to device for motion compensation
		cudaMemcpyAsync(&d_mbsY[split_atY*cm_gpu->mb_cols], &cm_cpu->mbs[0][split_atY*cm_gpu->mb_cols], cpu_partY*cm_gpu->mb_cols*sizeof(struct macroblock), cudaMemcpyHostToDevice, kernel_configs.motion_comp_Y.stream);
		cudaMemcpyAsync(&d_mbsU[split_atUV*(cm_gpu->mb_cols/2)], &cm_cpu->mbs[1][split_atUV*(cm_gpu->mb_cols/2)], cpu_partUV*cm_gpu->mb_cols/2*sizeof(struct macroblock), cudaMemcpyHostToDevice, kernel_configs.motion_comp_U.stream);
		cudaMemcpyAsync(&d_mbsV[split_atUV*(cm_gpu->mb_cols/2)], &cm_cpu->mbs[2][split_atUV*(cm_gpu->mb_cols/2)], cpu_partUV*cm_gpu->mb_cols/2*sizeof(struct macroblock), cudaMemcpyHostToDevice, kernel_configs.motion_comp_V.stream);
#endif
		c63_motion_compensate_Y_GPU<<<kernel_configs.motion_comp_Y.grid_size, kernel_configs.motion_comp_Y.block_size, 0, kernel_configs.motion_comp_Y.stream>>>
		(d_predictedY, d_ref_reconsY, d_mbsY, cm_gpu->width, cm_gpu->mb_cols, cm_gpu->mb_rows);
		c63_motion_compensate_U_GPU<<<kernel_configs.motion_comp_U.grid_size, kernel_configs.motion_comp_U.block_size, 0, kernel_configs.motion_comp_U.stream>>>
		(d_predictedU, d_ref_reconsU, d_mbsU, cm_gpu->width / 2, cm_gpu->mb_cols / 2, cm_gpu->mb_rows / 2);
		c63_motion_compensate_V_GPU<<<kernel_configs.motion_comp_V.grid_size, kernel_configs.motion_comp_V.block_size, 0, kernel_configs.motion_comp_V.stream>>>
		(d_predictedV, d_ref_reconsV, d_mbsV, cm_gpu->width / 2, cm_gpu->mb_cols / 2, cm_gpu->mb_rows / 2);
	}

	int i;
	for (i = 0; i < kernel_configs.n_chunks; ++i)
	{
		int offsetY = i * kernel_configs.quantdequant_Y.size_chunk;
		int offsetU = i * kernel_configs.quantdequant_U.size_chunk;
		int offsetV = i * kernel_configs.quantdequant_V.size_chunk;

		dct_quantize_GPU<<<kernel_configs.quantdequant_Y.grid_size, kernel_configs.quantdequant_Y.block_size, 0, kernel_configs.quantdequant_Y.stream>>>
		(&d_origY[offsetY], &d_predictedY[offsetY], cm_gpu->ypw, &d_residualsYDCT[offsetY], 0);
		cudaMemcpyAsync(&cm_cpu->residuals->Ydct[offsetY], &d_residualsYDCT[offsetY], kernel_configs.quantdequant_Y.size_chunk * sizeof(int16_t), cudaMemcpyDeviceToHost, kernel_configs.quantdequant_Y.stream);

		dct_quantize_GPU<<<kernel_configs.quantdequant_U.grid_size, kernel_configs.quantdequant_U.block_size, 0, kernel_configs.quantdequant_U.stream>>>
		(&d_origU[offsetU], &d_predictedU[offsetU], cm_gpu->upw, &d_residualsUDCT[offsetU], 1);
		cudaMemcpyAsync(&cm_cpu->residuals->Udct[offsetU], &d_residualsUDCT[offsetU], kernel_configs.quantdequant_U.size_chunk * sizeof(int16_t), cudaMemcpyDeviceToHost, kernel_configs.quantdequant_U.stream);

		dct_quantize_GPU<<<kernel_configs.quantdequant_V.grid_size, kernel_configs.quantdequant_V.block_size, 0, kernel_configs.quantdequant_V.stream>>>
		(&d_origV[offsetV], &d_predictedV[offsetV], cm_gpu->vpw, &d_residualsVDCT[offsetV], 1);
		cudaMemcpyAsync(&cm_cpu->residuals->Vdct[offsetV], &d_residualsVDCT[offsetV], kernel_configs.quantdequant_V.size_chunk * sizeof(int16_t), cudaMemcpyDeviceToHost, kernel_configs.quantdequant_V.stream);
	}

	for (i = 0; i < kernel_configs.n_chunks; ++i)
	{
		int offsetY = i * kernel_configs.quantdequant_Y.size_chunk;
		int offsetU = i * kernel_configs.quantdequant_U.size_chunk;
		int offsetV = i * kernel_configs.quantdequant_V.size_chunk;

		dequantize_idct_GPU<<<kernel_configs.quantdequant_Y.grid_size, kernel_configs.quantdequant_Y.block_size, 0, kernel_configs.quantdequant_Y.stream>>>
		(&d_residualsYDCT[offsetY], &d_predictedY[offsetY], cm_gpu->ypw, &d_current_reconsY[offsetY], 0);

		dequantize_idct_GPU<<<kernel_configs.quantdequant_U.grid_size, kernel_configs.quantdequant_U.block_size, 0, kernel_configs.quantdequant_U.stream>>>
		(&d_residualsUDCT[offsetU], &d_predictedU[offsetU], cm_gpu->upw, &d_current_reconsU[offsetU], 1);

		dequantize_idct_GPU<<<kernel_configs.quantdequant_V.grid_size, kernel_configs.quantdequant_V.block_size, 0, kernel_configs.quantdequant_V.stream>>>
		(&d_residualsVDCT[offsetV], &d_predictedV[offsetV], cm_gpu->vpw, &d_current_reconsV[offsetV], 1);
	}

	/* Function dump_image(), found in common.c, can be used here to check if the
	prediction is correct */

#if USE_HETEROGENEOUS_PROCESSING
	//CPU writes these to the .c63 file so we have to transfer them back to the CPUp
	cudaMemcpyAsync(&cm_cpu->mbs[0][0], &d_mbsY[0], gpu_partY*cm_gpu->mb_cols*sizeof(struct macroblock), cudaMemcpyDeviceToHost, kernel_configs.motion_comp_Y.stream);
	cudaMemcpyAsync(&cm_cpu->mbs[1][0], &d_mbsU[0], gpu_partUV*cm_gpu->mb_cols/2*sizeof(struct macroblock), cudaMemcpyDeviceToHost, kernel_configs.motion_comp_U.stream);
	cudaMemcpy(&cm_cpu->mbs[2][0], &d_mbsV[0], gpu_partUV*cm_gpu->mb_cols/2*sizeof(struct macroblock), cudaMemcpyDeviceToHost);
#else
	cudaMemcpyAsync(cm_cpu->mbs[0], d_mbsY, cm_gpu->mb_rows*cm_gpu->mb_cols*sizeof(struct macroblock), cudaMemcpyDeviceToHost, kernel_configs.motion_comp_Y.stream);
	cudaMemcpyAsync(cm_cpu->mbs[1], d_mbsU, cm_gpu->mb_rows/2 * cm_gpu->mb_cols/2 * sizeof(struct macroblock), cudaMemcpyDeviceToHost, kernel_configs.motion_comp_U.stream);
	cudaMemcpy(cm_cpu->mbs[2], d_mbsV, cm_gpu->mb_rows/2 * cm_gpu->mb_cols/2 * sizeof(struct macroblock), cudaMemcpyDeviceToHost);
#endif

	write_frame(cm_cpu);

	++cm_gpu->framenum;
	++cm_gpu->frames_since_keyframe;

	final_time += me_time;
}

struct c63_common_gpu* init_c63_enc_gpu(int width, int height)
{
  int i;

  /* calloc() sets allocated memory to zero */
  struct c63_common_gpu *cm = (c63_common_gpu*) calloc(1, sizeof(struct c63_common_gpu));

  cm->width = width;
  cm->height = height;

  cm->padw[0] = cm->ypw = (uint32_t)(ceil(width/16.0f)*16);
  cm->padh[0] = cm->yph = (uint32_t)(ceil(height/16.0f)*16);
  cm->padw[1] = cm->upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
  cm->padh[1] = cm->uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
  cm->padw[2] = cm->vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
  cm->padh[2] = cm->vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

  cm->mb_cols = cm->ypw / 8;
  cm->mb_rows = cm->yph / 8;

  /* Quality parameters */
  cm->qp = 25;                  // Constant quantization factor. Range: [1..50]
  cm->me_search_range = 16;     // Pixels in every direction
  cm->keyframe_interval = 100;  // Distance between keyframes

  /* Initialize quantization tables */
  for (i = 0; i < 64; ++i)
  {
    cm->quanttbl[0][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[1][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[2][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  return cm;
}

struct c63_common_cpu* init_c63_enc_cpu(int width, int height)
{
  int i;

  /* calloc() sets allocated memory to zero */
  struct c63_common_cpu *cm = (c63_common_cpu*) calloc(1, sizeof(struct c63_common_cpu));

  cm->width = width;
  cm->height = height;

  cm->padw[0] = cm->ypw = (uint32_t)(ceil(width/16.0f)*16);
  cm->padh[0] = cm->yph = (uint32_t)(ceil(height/16.0f)*16);
  cm->padw[1] = cm->upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
  cm->padh[1] = cm->uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
  cm->padw[2] = cm->vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
  cm->padh[2] = cm->vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

  cm->mb_cols = cm->ypw / 8;
  cm->mb_rows = cm->yph / 8;

  /* Quality parameters */
  cm->qp = 25;                  // Constant quantization factor. Range: [1..50]
  cm->me_search_range = 16;     // Pixels in every direction
  cm->keyframe_interval = 100;  // Distance between keyframes

  /* Initialize quantization tables */
  for (i = 0; i < 64; ++i)
  {
    cm->quanttbl[0][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[1][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[2][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  cm->ref_recons = (yuv_t*) malloc(sizeof(yuv_t));
  cm->ref_recons->Y = (uint8_t*) malloc(cm->ypw * cm->yph);
  cm->ref_recons->U = (uint8_t*) malloc(cm->upw * cm->uph);
  cm->ref_recons->V = (uint8_t*) malloc(cm->vpw * cm->vph);

  cm->residuals = (dct_t*) malloc(sizeof(dct_t));
  cudaMallocHost((void**)&cm->residuals->Ydct, cm->ypw * cm->yph * sizeof(int16_t));
  cudaMallocHost((void**)&cm->residuals->Udct, cm->upw * cm->uph * sizeof(int16_t));
  cudaMallocHost((void**)&cm->residuals->Vdct, cm->vpw * cm->vph * sizeof(int16_t));

  cudaMallocHost((void**)&cm->mbs[0], cm->mb_rows * cm->mb_cols * sizeof(struct macroblock));
  cudaMallocHost((void**)&cm->mbs[1], cm->mb_rows/2 * cm->mb_cols/2 * sizeof(struct macroblock));
  cudaMallocHost((void**)&cm->mbs[2], cm->mb_rows/2 * cm->mb_cols/2 * sizeof(struct macroblock));

  return cm;
}

static void print_help()
{
  printf("%s\n", "Usage: ./c63enc [options] input_file");
  printf("%s\n", "Commandline options:");
  printf("%s\n", "  -h                             Height of images to compress");
  printf("%s\n", "  -w                             Width of images to compress");
  printf("%s\n", "  -o                             Output file (.c63)");
  printf("%s\n", "  [-f]                           Limit number of frames to encode");
  printf("%s\n", "\n");

  exit(EXIT_FAILURE);
}

#ifndef WIN32
int main(int argc, char **argv)
{
  int c;
  struct macroblock *d_mbsY = 0;
  struct macroblock *d_mbsU = 0;
  struct macroblock *d_mbsV = 0;
  uint8_t *d_origY = 0;
  uint8_t *d_origU = 0;
  uint8_t *d_origV = 0;
  uint8_t *d_current_reconsY = 0;
  uint8_t *d_current_reconsU = 0;
  uint8_t *d_current_reconsV = 0;
  uint8_t *d_ref_reconsY = 0;
  uint8_t *d_ref_reconsU = 0;
  uint8_t *d_ref_reconsV = 0;
  uint8_t *d_predictedY = 0;
  uint8_t *d_predictedU = 0;
  uint8_t *d_predictedV = 0;
  int16_t *d_residualsYDCT = 0;
  int16_t *d_residualsUDCT = 0;
  int16_t *d_residualsVDCT = 0;
  uint8_t *d_quant0 = 0;
  uint8_t *d_quant1 = 0;
  uint8_t *d_quant2 = 0;

  if (argc == 1) { print_help(); }

  while ((c = getopt(argc, argv, "h:w:o:f:i:")) != -1)
  {
    switch (c)
    {
      case 'h':
        height = atoi(optarg);
        break;
      case 'w':
        width = atoi(optarg);
        break;
      case 'o':
        output_file = optarg;
        break;
      case 'f':
        limit_numframes = atoi(optarg);
        break;
      default:
        print_help();
        break;
    }
  }

  if (optind >= argc)
  {
    fprintf(stderr, "Error getting program options, try --help.\n");
    exit(EXIT_FAILURE);
  }

  outfile = fopen(output_file, "wb");

  if (outfile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  struct c63_common_cpu *cm_cpu = init_c63_enc_cpu(width, height);
  cm_cpu->e_ctx.fp = outfile;
  struct c63_common_gpu *cm_gpu = init_c63_enc_gpu(width, height);
  cm_gpu->e_ctx.fp = outfile;
  //print_GPU_info();

  input_file = argv[optind];
  
  struct KernelConfigurations kernel_configs = init_kernel_configurations(width, height, cm_cpu);
  partition_domain(width, height, cm_cpu);

  if (limit_numframes) { printf("Limited to %d frames.\n", limit_numframes); }

  FILE *infile = fopen(input_file, "rb");

  if (infile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  // Encode input frames 
  int numframes = 0;

  cudaEventCreate(&start_event);
  cudaEventCreate(&end_event);

  double timeSpent;
  long startTime = clock();

  //Allocate GPU Data
  cudaMalloc((void**)&d_ref_reconsY, cm_cpu->ypw * cm_cpu->yph);
  cudaMalloc((void**)&d_ref_reconsU, cm_cpu->upw * cm_cpu->uph);
  cudaMalloc((void**)&d_ref_reconsV, cm_cpu->vpw * cm_cpu->vph);

  cudaMalloc((void**)&d_current_reconsY, cm_cpu->ypw * cm_cpu->yph);
  cudaMalloc((void**)&d_current_reconsU, cm_cpu->upw * cm_cpu->uph);
  cudaMalloc((void**)&d_current_reconsV, cm_cpu->vpw * cm_cpu->vph);

  cudaMalloc((void**)&d_mbsY, cm_cpu->mb_rows * cm_cpu->mb_cols * sizeof(struct macroblock));
  cudaMalloc((void**)&d_mbsU, cm_cpu->mb_rows/2 * cm_cpu->mb_cols/2 * sizeof(struct macroblock));
  cudaMalloc((void**)&d_mbsV, cm_cpu->mb_rows/2 * cm_cpu->mb_cols/2 * sizeof(struct macroblock));

  cudaMalloc((void**)&d_origY, cm_cpu->padw[0]*cm_cpu->padh[0]);
  cudaMalloc((void**)&d_origU, cm_cpu->padw[1]*cm_cpu->padh[1]);
  cudaMalloc((void**)&d_origV, cm_cpu->padw[2]*cm_cpu->padh[2]);

  cudaMalloc((void**)&d_predictedY, cm_cpu->ypw * cm_cpu->yph);
  cudaMalloc((void**)&d_predictedU, cm_cpu->upw * cm_cpu->uph);
  cudaMalloc((void**)&d_predictedV, cm_cpu->vpw * cm_cpu->vph);

  cudaMalloc((void**)&d_residualsYDCT, cm_cpu->ypw * cm_cpu->yph * sizeof(int16_t));
  cudaMalloc((void**)&d_residualsUDCT, cm_cpu->upw * cm_cpu->uph * sizeof(int16_t));
  cudaMalloc((void**)&d_residualsVDCT, cm_cpu->vpw * cm_cpu->vph * sizeof(int16_t));

  cudaMemset((void*)d_predictedY, 0, cm_cpu->ypw * cm_cpu->yph * sizeof(uint8_t));
  cudaMemset((void*)d_predictedU, 0, cm_cpu->upw * cm_cpu->uph * sizeof(uint8_t));
  cudaMemset((void*)d_predictedV, 0, cm_cpu->vpw * cm_cpu->vph * sizeof(uint8_t));

  cudaMemset((void*)d_residualsYDCT, 0, cm_cpu->ypw * cm_cpu->yph * sizeof(int16_t));
  cudaMemset((void*)d_residualsUDCT, 0, cm_cpu->upw * cm_cpu->uph * sizeof(int16_t));
  cudaMemset((void*)d_residualsVDCT, 0, cm_cpu->vpw * cm_cpu->vph * sizeof(int16_t));

  while (1)
  {
    cm_cpu->orig = read_yuv(infile, cm_cpu);

	if (!cm_cpu->orig) { break; }

    printf("Encoding frame %d, ", numframes);
	c63_encode_image(cm_cpu, cm_gpu, kernel_configs, d_origY, d_origU, d_origV, d_ref_reconsY, d_ref_reconsU, d_ref_reconsV, d_current_reconsY, d_current_reconsU, d_current_reconsV, 
					d_predictedY, d_predictedU, d_predictedV, d_mbsY, d_mbsU, d_mbsV, d_residualsYDCT, d_residualsUDCT, d_residualsVDCT, d_quant0, d_quant1, d_quant2);

	cudaFreeHost((void*)cm_cpu->orig->Y);
	cudaFreeHost((void*)cm_cpu->orig->U);
	cudaFreeHost((void*)cm_cpu->orig->V);
    free(cm_cpu->orig);

    printf("%s\n", "Done!");

    ++numframes;

    if (limit_numframes && numframes >= limit_numframes) { break; }
  }

 //Free GPU data
  cudaFree((void*)d_ref_reconsY);
  cudaFree((void*)d_ref_reconsU);
  cudaFree((void*)d_ref_reconsV);

  cudaFree((void*)d_current_reconsY);
  cudaFree((void*)d_current_reconsU);
  cudaFree((void*)d_current_reconsV);

  cudaFree((void*)d_mbsY);
  cudaFree((void*)d_mbsU);
  cudaFree((void*)d_mbsV);

  cudaFree((void*)d_origY);
  cudaFree((void*)d_origU);
  cudaFree((void*)d_origV);

  cudaFree((void*)d_predictedY);
  cudaFree((void*)d_predictedU);
  cudaFree((void*)d_predictedV);

  cudaFree((void*)d_residualsYDCT);
  cudaFree((void*)d_residualsUDCT);
  cudaFree((void*)d_residualsVDCT);

  //Free CPU data
  free(cm_cpu->ref_recons->Y);
  free(cm_cpu->ref_recons->U);
  free(cm_cpu->ref_recons->V);
  free(cm_cpu->ref_recons);

  cudaFreeHost((void*)cm_cpu->residuals->Ydct);
  cudaFreeHost((void*)cm_cpu->residuals->Udct);
  cudaFreeHost((void*)cm_cpu->residuals->Vdct);
  free(cm_cpu->residuals);

  cudaFreeHost((void*)cm_cpu->mbs[0]);
  cudaFreeHost((void*)cm_cpu->mbs[1]);
  cudaFreeHost((void*)cm_cpu->mbs[2]);

  cudaEventDestroy(start_event);
  cudaEventDestroy(end_event);

  int i;
  for(i = 0; i < 3; ++i)
  {
	  cudaStreamDestroy(streams[i]);
  }

  for(i = 0; i < kernel_configs.n_chunks; ++i)
  {
	  cudaStreamDestroy(quantdeq_streamY[i]);
	  cudaStreamDestroy(quantdeq_streamU[i]);
	  cudaStreamDestroy(quantdeq_streamV[i]);
  }

  long endTime = clock();
  timeSpent = (double)(endTime - startTime) / CLOCKS_PER_SEC;

  printf("Clock Time: %f\n", timeSpent);
  printf("Final CUDA kernel time: %f\n", final_time);

  fclose(outfile);
  fclose(infile);
  
  cudaDeviceReset();

  return EXIT_SUCCESS;
}
#else
int main(int argc, char **argv)
{
  int c;
  struct macroblock *d_mbsY = 0;
  struct macroblock *d_mbsU = 0;
  struct macroblock *d_mbsV = 0;
  uint8_t *d_origY = 0;
  uint8_t *d_origU = 0;
  uint8_t *d_origV = 0;
  uint8_t *d_current_reconsY = 0;
  uint8_t *d_current_reconsU = 0;
  uint8_t *d_current_reconsV = 0;
  uint8_t *d_ref_reconsY = 0;
  uint8_t *d_ref_reconsU = 0;
  uint8_t *d_ref_reconsV = 0;
  uint8_t *d_predictedY = 0;
  uint8_t *d_predictedU = 0;
  uint8_t *d_predictedV = 0;
  int16_t *d_residualsYDCT = 0;
  int16_t *d_residualsUDCT = 0;
  int16_t *d_residualsVDCT = 0;
  uint8_t *d_quant0 = 0;
  uint8_t *d_quant1 = 0;
  uint8_t *d_quant2 = 0;

  if (argc == 1) { print_help(); }

  width = atoi(argv[1]);
  height = atoi(argv[2]);
  output_file = argv[3];

  outfile = fopen(output_file, "wb");

  if (outfile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  struct c63_common_cpu *cm_cpu = init_c63_enc_cpu(width, height);
  cm_cpu->e_ctx.fp = outfile;
  struct c63_common_gpu *cm_gpu = init_c63_enc_gpu(width, height);
  cm_gpu->e_ctx.fp = outfile;
  //print_GPU_info();

  input_file = argv[4];
  
  struct KernelConfigurations kernel_configs = init_kernel_configurations(width, height, cm_cpu);
  partition_domain(width, height, cm_cpu);

  if (limit_numframes) { printf("Limited to %d frames.\n", limit_numframes); }

  FILE *infile = fopen(input_file, "rb");

  if (infile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  // Encode input frames 
  int numframes = 0;

  cudaEventCreate(&start_event);
  cudaEventCreate(&end_event);

  double timeSpent;
  long startTime = clock();

  //Allocate GPU Data
  cudaMalloc((void**)&d_ref_reconsY, cm_cpu->ypw * cm_cpu->yph);
  cudaMalloc((void**)&d_ref_reconsU, cm_cpu->upw * cm_cpu->uph);
  cudaMalloc((void**)&d_ref_reconsV, cm_cpu->vpw * cm_cpu->vph);

  cudaMalloc((void**)&d_current_reconsY, cm_cpu->ypw * cm_cpu->yph);
  cudaMalloc((void**)&d_current_reconsU, cm_cpu->upw * cm_cpu->uph);
  cudaMalloc((void**)&d_current_reconsV, cm_cpu->vpw * cm_cpu->vph);

  cudaMalloc((void**)&d_mbsY, cm_cpu->mb_rows * cm_cpu->mb_cols * sizeof(struct macroblock));
  cudaMalloc((void**)&d_mbsU, cm_cpu->mb_rows/2 * cm_cpu->mb_cols/2 * sizeof(struct macroblock));
  cudaMalloc((void**)&d_mbsV, cm_cpu->mb_rows/2 * cm_cpu->mb_cols/2 * sizeof(struct macroblock));

  cudaMalloc((void**)&d_origY, cm_cpu->padw[0]*cm_cpu->padh[0]);
  cudaMalloc((void**)&d_origU, cm_cpu->padw[1]*cm_cpu->padh[1]);
  cudaMalloc((void**)&d_origV, cm_cpu->padw[2]*cm_cpu->padh[2]);

  cudaMalloc((void**)&d_predictedY, cm_cpu->ypw * cm_cpu->yph);
  cudaMalloc((void**)&d_predictedU, cm_cpu->upw * cm_cpu->uph);
  cudaMalloc((void**)&d_predictedV, cm_cpu->vpw * cm_cpu->vph);

  cudaMalloc((void**)&d_residualsYDCT, cm_cpu->ypw * cm_cpu->yph * sizeof(int16_t));
  cudaMalloc((void**)&d_residualsUDCT, cm_cpu->upw * cm_cpu->uph * sizeof(int16_t));
  cudaMalloc((void**)&d_residualsVDCT, cm_cpu->vpw * cm_cpu->vph * sizeof(int16_t));

  cudaMemset((void*)d_predictedY, 0, cm_cpu->ypw * cm_cpu->yph * sizeof(uint8_t));
  cudaMemset((void*)d_predictedU, 0, cm_cpu->upw * cm_cpu->uph * sizeof(uint8_t));
  cudaMemset((void*)d_predictedV, 0, cm_cpu->vpw * cm_cpu->vph * sizeof(uint8_t));

  cudaMemset((void*)d_residualsYDCT, 0, cm_cpu->ypw * cm_cpu->yph * sizeof(int16_t));
  cudaMemset((void*)d_residualsUDCT, 0, cm_cpu->upw * cm_cpu->uph * sizeof(int16_t));
  cudaMemset((void*)d_residualsVDCT, 0, cm_cpu->vpw * cm_cpu->vph * sizeof(int16_t));

  while (1)
  {
    cm_cpu->orig = read_yuv(infile, cm_cpu);

	if (!cm_cpu->orig) { break; }

    printf("Encoding frame %d, ", numframes);
	c63_encode_image(cm_cpu, cm_gpu, kernel_configs, d_origY, d_origU, d_origV, d_ref_reconsY, d_ref_reconsU, d_ref_reconsV, d_current_reconsY, d_current_reconsU, d_current_reconsV, 
					d_predictedY, d_predictedU, d_predictedV, d_mbsY, d_mbsU, d_mbsV, d_residualsYDCT, d_residualsUDCT, d_residualsVDCT, d_quant0, d_quant1, d_quant2);

	cudaFreeHost((void*)cm_cpu->orig->Y);
	cudaFreeHost((void*)cm_cpu->orig->U);
	cudaFreeHost((void*)cm_cpu->orig->V);
    free(cm_cpu->orig);

    printf("%s\n", "Done!");

    ++numframes;

    if (limit_numframes && numframes >= limit_numframes) { break; }
  }

  //Free GPU data
  cudaFree((void*)d_ref_reconsY);
  cudaFree((void*)d_ref_reconsU);
  cudaFree((void*)d_ref_reconsV);

  cudaFree((void*)d_current_reconsY);
  cudaFree((void*)d_current_reconsU);
  cudaFree((void*)d_current_reconsV);

  cudaFree((void*)d_mbsY);
  cudaFree((void*)d_mbsU);
  cudaFree((void*)d_mbsV);

  cudaFree((void*)d_origY);
  cudaFree((void*)d_origU);
  cudaFree((void*)d_origV);

  cudaFree((void*)d_predictedY);
  cudaFree((void*)d_predictedU);
  cudaFree((void*)d_predictedV);

  cudaFree((void*)d_residualsYDCT);
  cudaFree((void*)d_residualsUDCT);
  cudaFree((void*)d_residualsVDCT);

  //Free CPU data
  free(cm_cpu->ref_recons->Y);
  free(cm_cpu->ref_recons->U);
  free(cm_cpu->ref_recons->V);
  free(cm_cpu->ref_recons);

  cudaFreeHost((void*)cm_cpu->residuals->Ydct);
  cudaFreeHost((void*)cm_cpu->residuals->Udct);
  cudaFreeHost((void*)cm_cpu->residuals->Vdct);
  free(cm_cpu->residuals);

  cudaFreeHost((void*)cm_cpu->mbs[0]);
  cudaFreeHost((void*)cm_cpu->mbs[1]);
  cudaFreeHost((void*)cm_cpu->mbs[2]);

  cudaEventDestroy(start_event);
  cudaEventDestroy(end_event);

  for(int i = 0; i < 3; ++i)
  {
	  cudaStreamDestroy(streams[i]);
  }

  for(int i = 0; i < kernel_configs.n_chunks; ++i)
  {
	  cudaStreamDestroy(quantdeq_streamY[i]);
	  cudaStreamDestroy(quantdeq_streamU[i]);
	  cudaStreamDestroy(quantdeq_streamV[i]);
  }

  long endTime = clock();
  timeSpent = (double)(endTime - startTime) / CLOCKS_PER_SEC;

  printf("Clock Time: %f\n", timeSpent);
  printf("Final CUDA kernel time: %f\n", final_time);

  fclose(outfile);
  fclose(infile);

  cudaDeviceReset();
  
  system("pause");
  return EXIT_SUCCESS;
}
#endif
