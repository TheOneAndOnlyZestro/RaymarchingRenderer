#ifndef FKA
#define FKA

#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <filesystem>
#include <assert.h>
#include <sstream>
#include "math.h"
#include "ray_maths.cuh"
#include "ray_FFT.cuh"
#include <cuda_runtime.h>

void launchFragment(cudaSurfaceObject_t surf, float time, unsigned int width, unsigned int height,const ray::vec3& loc,const ray::vec3& rot, const float n);

#endif



