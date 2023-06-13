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
	for (int i=0; i<4; i++){
		bbox[i] = q17_14_to_float(ml_data[i]);
	}
}

float q17_14_to_float(uint32_t q_value) {
        // Extract the sign bit
   uint32_t sign = (q_value >> 31) & 0x1;

   if (sign==1)
    q_value = (~q_value) + 1;

        // Extract the integer part
     uint32_t integer_part = (q_value >> 14) & 0x1FFFF;

        // Extract the fractional part
     uint32_t fractional_part = q_value & 0x3FFF;

        // Calculate the float value
        //float float_value = (float)integer_part + (float)fractional_part / 16384.0f; //16384.0f = 2**14
     float float_value = (float)fractional_part / 16384.0f;


        // Apply the sign
        if (sign==1){
            float_value *= -1.0f;
        }

        //printf("%f", float_value);

        return float_value;
}
