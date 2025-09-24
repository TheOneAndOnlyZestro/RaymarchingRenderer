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

void launchFragment(cudaSurfaceObject_t surf,unsigned int width, unsigned int height, float time, const float* output, const unsigned int output_size ,
const PrimitiveType* output_disc,const unsigned int outputDiscSize, const ray::vec3* lightSource);
void Allocate(float** output_device, PrimitiveType** output_disc_device, unsigned int output_size, unsigned int outputDiscSize);
void Free(float* output_device, PrimitiveType* output_disc_device);
void toDevice(float* output_host, PrimitiveType* output_disc_host,float* output_device, PrimitiveType* output_disc_device, unsigned int output_size, unsigned int outputDiscSize);
void AllocateLightSource(ray::vec3** LightSourceDevice);
void UpdateDeviceLightSource(const ray::vec3* lightSource, ray::vec3* LightSourceDevice);
void FreeDeviceLightSource(ray::vec3* LightSourceDevice);


#endif



