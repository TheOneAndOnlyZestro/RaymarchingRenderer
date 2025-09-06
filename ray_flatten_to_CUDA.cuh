//
// Created by Andrew on 9/6/2025.
//

#ifndef RAYMARCHINGCUDA_RAY_FLATTEN_TO_CUDA_H
#define RAYMARCHINGCUDA_RAY_FLATTEN_TO_CUDA_H


#include "Primitive.cuh"
#include <vector>

#define MAX_PARAMS 32
namespace ray {
    //Flattens all data of an object to an array of floats of all data in postfix notation
    __host__
    void flatten(const std::shared_ptr<Primitive>& object, std::vector<float>* out, std::vector<PrimitiveType>* outDesc);
}
#endif //RAYMARCHINGCUDA_RAY_FLATTEN_TO_CUDA_H