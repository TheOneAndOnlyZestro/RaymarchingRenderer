//
// Created by Andrew on 9/25/2025.
//

#ifndef RAYMARCHINGCUDA_PRIMITIVEUTILS_CUH
#include <cuda_runtime.h>

namespace PrimitiveUtils {
    enum class PrimitiveType {
        PRIMITIVE,
        SPHERE, CUBE, MANDELBROT, LINE, //Objects
        BINARY_OPERATOR,
        UNION, INTERSECT    //Operators
    };

    __device__ __host__
    inline const char* DebugPrim(PrimitiveType p) {
        switch (p) {
            case PrimitiveType::PRIMITIVE:
                return "PrimitiveType::PRIMITIVE";
            case PrimitiveType::CUBE:
                return "PrimitiveType::CUBE";
            case PrimitiveType::SPHERE:
                return "PrimitiveType::SPHERE";
            case PrimitiveType::LINE:
                return "PrimitiveType::LINE";
            case PrimitiveType::BINARY_OPERATOR:
                return "PrimitiveType::BINARY_OPERATOR";
            case PrimitiveType::MANDELBROT:
                return "PrimitiveType::MANDELBROT";
            case PrimitiveType::UNION:
                return "PrimitiveType::UNION";
            case PrimitiveType::INTERSECT:
                return "PrimitiveType::INTERSECT";
            default:
                return "NOT::PRIMITIVE";
        }
    }
    __device__ __host__
    inline size_t getPrimSize(PrimitiveType p) {
        switch (p) {
            case PrimitiveType::PRIMITIVE:
                return 9 +1;
            case PrimitiveType::CUBE:
                return 9 + 1;
            case PrimitiveType::SPHERE:
                return 10 + 1;
            case PrimitiveType::MANDELBROT:
                return 11 + 1;
            case PrimitiveType::LINE:
                return 9 + 3*2 + 1 + 1;
            case PrimitiveType::BINARY_OPERATOR:
                return 9 + 1;
            case PrimitiveType::UNION:
                return 9 + 1;
            case PrimitiveType::INTERSECT:
                return 9 + 1;
            default:
                return 0;
        }
    }
}
#define RAYMARCHINGCUDA_PRIMITIVEUTILS_CUH

#endif //RAYMARCHINGCUDA_PRIMITIVEUTILS_CUH