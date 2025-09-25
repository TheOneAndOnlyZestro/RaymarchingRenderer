//
// Created by Andrew on 9/25/2025.
//

#ifndef RAYMARCHINGCUDA_PRIMITIVEDEVICE_CUH

#include "ray_maths.cuh"
#include <cuda_runtime.h>
namespace SDF {
    __device__
    inline void Sphere(const ray::vec3& p, const float* input, size_t* size, float* out) {
        //Extract Featuress
        const ray::vec3 loc = &input[0];
        const ray::vec3 rot = &input[3];
        const ray::vec3 scale = &input[6];
        const float radius = input[9];

        if (size != nullptr) *size = 1;
        out[0]=ray::length(ray::ApplyTransform(p,loc,rot)) - (radius);
    }
    __device__
    inline void Cube(const ray::vec3& p, const float* input, size_t* size, float* out) {
        //Extract Features
        const ray::vec3 loc = &input[0];
        const ray::vec3 rot = &input[3];
        const ray::vec3 scale = &input[6];

        if (size != nullptr) *size = 1;
        ray::vec3 q(abs( ray::ApplyTransform(p,loc,rot)) - scale);
        out[0] = ray::length(ray::max(q, 0.f)) + fminf( ray::compMax(q) , 0.0) - 0.01f;
    }

    __device__
    inline void Mandelbulb(const ray::vec3& p, const float* input, size_t* size, float* out) {
        //Extract Features
        const ray::vec3 loc = &input[0];
        const ray::vec3 rot = &input[3];
        const ray::vec3 scale = &input[6];
        const unsigned int iterations = static_cast<unsigned int>(input[9]);
        const float exponent = input[10];

        ray::vec3 pnew = ray::ApplyTransform(p,loc,rot);
        ray::vec3 zold(0.f,0.f,0.f);
        ray::vec3 znew(0.f,0.f,0.f);

        float dr = 1.0f;
        for (unsigned int i = 0; i < iterations; i++) {
            if (ray::length(zold) > 8.f) {
                break;
            }
            znew = (zold ^ exponent) + pnew;
            dr = (exponent * powf(length(zold), exponent-1.f) * dr) + 1.f;
            zold = znew;
        }

        if (size!=nullptr)
            *size = 2;
        out[0] = 0.5f * (ray::length(znew) * logf(ray::length(znew)) )/(dr+EPSILON);
        //float v = length(zold) - floorf(length(zold));
        //out[1] =  v;
    }

    __device__
    inline void Line(const ray::vec3& p, const float* input, size_t* size, float* out) {
        //Extract Features
        const ray::vec3 loc = &input[0];
        const ray::vec3 rot = &input[3];
        const ray::vec3 scale = &input[6];
        const ray::vec3 a = &input[9];
        const ray::vec3 b = &input[12];
        const float radius = input[15];

        ray::vec3 pa = ray::ApplyTransform(p,loc,rot) - a;
        ray::vec3 ba = b - a;
        float h = ray::clamp( ray::dot(pa,ba)/ray::dot(ba,ba), 0.0, 1.0 );
        out[0] = ray::length( pa - ba*h ) - radius;
        *size=1;
    }

}

namespace Operator {
    __device__
    inline void Union(const float d1, const float d2, size_t *size, float *out) {
        if (size != nullptr)
            *size = 4;
        out[0] = fminf(d1, d2);
    }

    __device__
    inline void Intersect(const float d1, const float d2, size_t *size, float *out) {
        if (size != nullptr)
            *size = 4;
        out[0] = fmaxf(d1, d2);
    }
}
#define RAYMARCHINGCUDA_PRIMITIVEDEVICE_CUH

#endif //RAYMARCHINGCUDA_PRIMITIVEDEVICE_CUH