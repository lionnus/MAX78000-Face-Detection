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
		bbox[i] = (q17_14_to_float(ml_data[i])+1)*24;
	}
}

float q17_14_to_float(uint32_t q_value) {
      // Extract the sign bit
      uint32_t sign = (q_value >> 31) & 0x1;
      if (sign) {
    	  q_value = (~q_value)+1;
      }

      // Extract the integer part
      uint32_t integer_part = (q_value >> 14) & 0x1FFFF;

      // Extract the fractional part
      uint32_t fractional_part = q_value & 0x3FFF;

      // Calculate the float value
      float float_value = (float)integer_part + (float)fractional_part / (float)(1<<14);

      // Apply the sign
      if (sign)
          float_value *= -1.0f;

      return float_value;
  }
