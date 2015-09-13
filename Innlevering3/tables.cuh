#pragma once

#ifdef WIN32
#include<cstdint>
#else
#include <inttypes.h>
#endif

#ifndef WIN32
extern uint8_t yquanttbl_def[64] __attribute__((aligned(16)));
extern uint8_t uvquanttbl_def[64] __attribute__((aligned(16)));
#else
__declspec(align(16)) extern uint8_t yquanttbl_def[64];
__declspec(align(16)) extern uint8_t uvquanttbl_def[64];
#endif
extern uint16_t DCVLC[2][12];
extern uint8_t DCVLC_Size[2][12];
extern uint8_t DCVLC_num_by_length[2][16];
extern uint8_t DCVLC_data[2][12];
extern uint16_t ACVLC[2][16][11];
extern uint8_t ACVLC_Size[2][16][11];
extern uint8_t ACVLC_num_by_length[2][16];
extern uint8_t ACVLC_data[2][162];
extern uint8_t zigzag_U[64];
extern uint8_t zigzag_V[64];
extern float dctlookup[8][8];
extern uint16_t MVVLC[8];
extern uint8_t MVVLC_Size[8];
