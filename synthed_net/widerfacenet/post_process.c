/******************************************************************************
 * file     post_process.c
 * brief    Post processing functions for the WiderFaceONet
 * version  V1.0
 * date     2. June 2023
 * 
 *
 ******************************************************************************/
#include "post_process.h"

void convert_to_bbox(uint32_t *ml_data, float *bbox){
	uint32_t MAX_UINT32 = (2^32)-1;
	const float scaling_factor = 48.0/MAX_UINT32;
	for (int i=0; i<4; i++){
		bbox[i] = (float)ml_data[i] * scaling_factor;
	}
}
void convert_to_confid(uint32_t* ml_data, float* face_confid){
}

