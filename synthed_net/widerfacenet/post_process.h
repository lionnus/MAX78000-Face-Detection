/******************************************************************************
 * file     post_process.h
 * brief    Post processing functions for the WiderFaceONet
 * version  V1.0
 * date     2. June 2023
 * 
 *
 ******************************************************************************/

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define SQUARE(x) ((x) * (x))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

void convert_to_bbox(uint32_t* ml_data, float* bbox);
void convert_to_confid(uint32_t* ml_data, float* face_confid);
