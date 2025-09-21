#ifndef FKA
#define FKA

#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <filesystem>
#include <assert.h>
#include <sstream>
#include "math.h"
#include "Primitive.cuh"
#include "ray_FFT.cuh"
#include "ray_Stack.cuh"
#include <cuda_runtime.h>

void launchFragment(cudaSurfaceObject_t surf,unsigned int width, unsigned int height, float time, const float* output, const unsigned int output_size ,const PrimitiveType* output_disc,const unsigned int outputDiscSize);

#endif



