//
// Created by Andrew on 9/25/2025.
//

#ifndef RAYMARCHINGCUDA_MATERIAL_CUH
class Material {
private:
    float IDs[20];
    unsigned int numOfObjects;

    float specularPower;
    float intensity;
    float ambient;
    float reflection_intensity;
    float IOR;
    float specularIntensity;

public:
    __device__ __host__
    Material();

    __device__ __host__
    void getSpecularPower() const;
    __device__ __host__
    void getIntensity() const;
    __device__ __host__
    void getAmbient() const;
    __device__ __host__
    void getReflectionIntensity() const;
    __device__ __host__
    void getIOR() const;
    __device__ __host__
    void getSpecularIntensity() const;

    float* getSpecularPowerRef();
    float* getIntensityRef();
    float* getAmbientRef();
    float* getReflectionIntensityRef();
    float* getIORRef();
    float* getSpecularIntensityRef();

};


#define RAYMARCHINGCUDA_MATERIAL_CUH

#endif //RAYMARCHINGCUDA_MATERIAL_CUH