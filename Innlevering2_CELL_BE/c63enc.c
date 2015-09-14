#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<time.h>

#include "c63.h"
#include "tables.h"

static char *output_file, *input_file;
FILE *outfile;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;

/* getopt */
extern int optind;
extern char *optarg;

static void copy_image(uint8_t *newImage, uint8_t *image, int mb_rows, int mb_cols, int width)
{
	int mb_x, mb_y;
	int i = 0;
	int count = 0;

	// Luma
	for (mb_y = 0; mb_y < mb_rows; ++mb_y)
	{
		for (mb_x = 0; mb_x < mb_cols; ++mb_x)
		{
			int left = mb_x * 8;
			int top = mb_y * 8;
			int bottom = top + 8;

			int x, y;
			y = top;
			x = left;

			for (y = top; y < bottom; ++y)
			{
				count++;
				int number = y*width+x;
				memcpy((void*)&newImage[i], (void*)&image[number], 8 * sizeof(uint8_t));
				i+= 8;
			}
		}
	}
}

static void copy_recons(uint8_t *newRecons, uint8_t *recons, int mb_rows, int mb_cols, int width)
{
	int mb_x, mb_y;
	int i = 0;

	// Luma
	for (mb_y = 0; mb_y < mb_rows; ++mb_y)
	{
		for (mb_x = 0; mb_x < mb_cols; ++mb_x)
		{
			int left = mb_x * 8;
			int top = mb_y * 8;
			int bottom = top + 8;

			int x, y;
			y = top;
			x = left;

			for (y = top; y < bottom; ++y)
			{
				memcpy((void*)&newRecons[y*width+x], (void*)&recons[i], 8 * sizeof(uint8_t));
				i+= 8;
			}
		}
	}
}

static yuv_t* recons_image(struct c63_common *cm, yuv_t* image)
{
	yuv_t *newImage = NULL;
	posix_memalign((void**)&newImage, 128, sizeof(struct yuv));
	posix_memalign((void**)&newImage->Y, 128, width*height);
	posix_memalign((void**)&newImage->U, 128, (width*height)/4);
	posix_memalign((void**)&newImage->V, 128, (width*height)/4);

	copy_image(newImage->Y, image->Y, cm->mb_rows, cm->mb_cols, cm->padw[0]);
	copy_image(newImage->U, image->U, cm->mb_rows / 2, cm->mb_cols / 2, cm->padw[1]);
	copy_image(newImage->V, image->V, cm->mb_rows / 2, cm->mb_cols / 2, cm->padw[2]);

	return newImage;
}

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static yuv_t* read_yuv(FILE *file)
{
  size_t len = 0;
  yuv_t *image = NULL;
  posix_memalign((void**)&image, 128, sizeof(*image));
  //image = (yuv_t*) memalign(128, sizeof(*image));

  /* Read Y. The size of Y is the same as the size of the image. */
  posix_memalign((void**)&image->Y, 128, width*height);
  //image->Y = (uint8_t*) memalign(128, width*height);
  len += fread(image->Y, 1, width*height, file);

  /* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y
     because (height/2)*(width/2) = (height*width)/4. */
  posix_memalign((void**)&image->U, 128, (width*height)/4);
  //image->U = (uint8_t*) memalign(128, (width*height)/4);
  len += fread(image->U, 1, (width*height)/4, file);

  /* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
  posix_memalign((void**)&image->V, 128, (width*height)/4);
  //image->V = (uint8_t*) memalign(128, (width*height)/4);
  len += fread(image->V, 1, (width*height)/4, file);

  if (ferror(file))
  {
    perror("ferror");
    exit(EXIT_FAILURE);
  }

  if (feof(file))
  {
    free(image->Y);
    free(image->U);
    free(image->V);
    free(image);

    return NULL;
  }
  else if (len != width*height*1.5)
  {
    fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
    fprintf(stderr, "Wrong input? (height: %d width: %d)\n", height, width);

    free(image->Y);
    free(image->U);
    free(image->V);
    free(image);

    return NULL;
  }

  return image;
}

static void c63_encode_image(struct c63_common *cm, yuv_t *image, yuv_t *newImage)
{
  /* Advance to next frame */
  destroy_frame(cm->refframe);
  cm->refframe = cm->curframe;
  cm->curframe = create_frame(cm, image, newImage);

  /* Check if keyframe */
  if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval)
  {
    cm->curframe->keyframe = 1;
    cm->frames_since_keyframe = 0;

    fprintf(stderr, " (keyframe) ");
  }
  else { cm->curframe->keyframe = 0; }

  if (!cm->curframe->keyframe)
  {
	// Motion Estimation
	c63_motion_estimate_alloc(cm);
    c63_motion_estimate(cm);
    c63_motion_estimate_dealloc(cm);

    // Motion Compensation
    c63_motion_compensate(cm);
  }

  dct_quantize(cm->curframe->newOrig->Y, cm->curframe->predicted->Y, cm->padw[0], cm->padh[0],
		  outY, cm->quanttbl[0]);
  dct_quantize(cm->curframe->newOrig->U, cm->curframe->predicted->U, cm->padw[1], cm->padh[1],
		  outU, cm->quanttbl[1]);
  dct_quantize(cm->curframe->newOrig->V, cm->curframe->predicted->V, cm->padw[2], cm->padh[2],
		  outV, cm->quanttbl[2]);

  /* Reconstruct frame for inter-prediction */
  dequantize_idct(outY, cm->curframe->predicted->Y,
      cm->ypw, cm->yph, cm->curframe->recons->Y, cm->quanttbl[0]);
  dequantize_idct(outU, cm->curframe->predicted->U,
      cm->upw, cm->uph, cm->curframe->recons->U, cm->quanttbl[1]);
  dequantize_idct(outV, cm->curframe->predicted->V,
      cm->vpw, cm->vph, cm->curframe->recons->V, cm->quanttbl[2]);

	copy_recons(cm->curframe->newRecons->Y, cm->curframe->recons->Y, cm->mb_rows, cm->mb_cols, cm->padw[0]);
	copy_recons(cm->curframe->newRecons->U, cm->curframe->recons->U, cm->mb_rows / 2, cm->mb_cols / 2, cm->padw[1]);
	copy_recons(cm->curframe->newRecons->V, cm->curframe->recons->V, cm->mb_rows / 2, cm->mb_cols / 2, cm->padw[2]);


  /* Function dump_image(), found in common.c, can be used here to check if the
     prediction is correct */

  write_frame(cm);

  ++cm->framenum;
  ++cm->frames_since_keyframe;
}

struct c63_common* init_c63_enc(int width, int height)
{
  int i;
  /* calloc() sets allocated memory to zero */
  struct c63_common *cm = calloc(1, sizeof(struct c63_common));

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

static void print_help()
{
  printf("Usage: ./c63enc [options] input_file\n");
  printf("Commandline options:\n");
  printf("  -h                             Height of images to compress\n");
  printf("  -w                             Width of images to compress\n");
  printf("  -o                             Output file (.c63)\n");
  printf("  [-f]                           Limit number of frames to encode\n");
  printf("\n");

  exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
  int c;
  yuv_t *image;
  yuv_t *newImage;

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

  struct c63_common *cm = init_c63_enc(width, height);
  cm->e_ctx.fp = outfile;

  input_file = argv[optind];

  if (limit_numframes) { printf("Limited to %d frames.\n", limit_numframes); }

  FILE *infile = fopen(input_file, "rb");

  if(infile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  /* Encode input frames */
  int numframes = 0;

  double timeSpent;
  long startTime = clock();
  while (1)
  {
    image = read_yuv(infile);
    if(image)
    	newImage = recons_image(cm, image);

    if (!image) { break; }

    printf("Encoding frame %d, ", numframes);
    c63_encode_image(cm, image, newImage);;

    free(image->Y);
    free(image->U);
    free(image->V);
    free(image);
    free(newImage->Y);
    free(newImage->U);
	free(newImage->V);
	free(newImage);

    printf("Done!\n");

    ++numframes;;

    if (limit_numframes && numframes >= limit_numframes) { break; }
  }
  long endTime = clock();
  timeSpent = (double)(endTime - startTime) / CLOCKS_PER_SEC;

  printf("Time: %f\n", timeSpent);

  fclose(outfile);
  fclose(infile);

  //int i, j;
  //for (i = 0; i < 2; ++i)
  //{
  //  printf("int freq[] = {");
  //  for (j = 0; j < ARRAY_SIZE(frequencies[i]); ++j)
  //  {
  //    printf("%d, ", frequencies[i][j]);
  //  }
  //  printf("};\n");
  //}

  return EXIT_SUCCESS;
}
