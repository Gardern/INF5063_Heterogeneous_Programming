#ifndef C63CPU
#define C63CPU

#include"c63.cuh"

struct frame
{
  yuv_t *orig;        // Original input image
  yuv_t *recons;      // Reconstructed image
  yuv_t *predicted;   // Predicted frame from intra-prediction

  dct_t *residuals;   // Difference between original image and predicted frame

  struct macroblock *mbs[3];
  int keyframe;
};

struct c63_common_cpu
{
  int width, height;
  int ypw, yph, upw, uph, vpw, vph;

  int padw[3], padh[3];

  int mb_cols, mb_rows;

  uint8_t qp;                         // Quality parameter

  int me_search_range;

  uint8_t quanttbl[3][64];

  yuv_t *orig;
  yuv_t *ref_recons;
  dct_t *residuals;
  struct macroblock *mbs[3];

  struct frame *refframe;
  struct frame *curframe;

  int keyframe;

  int framenum;

  int keyframe_interval;
  int frames_since_keyframe;

  struct entropy_ctx e_ctx;
};
void write_frame(struct c63_common_cpu *cm);
struct frame* create_frame(struct c63_common_cpu *cm, yuv_t *image);
void c63_motion_estimate_Y_CPU(struct c63_common_cpu *cm, int cpu_start);
void c63_motion_estimate_U_CPU(struct c63_common_cpu *cm, int cpu_start);
void c63_motion_estimate_V_CPU(struct c63_common_cpu *cm, int cpu_start);

#endif