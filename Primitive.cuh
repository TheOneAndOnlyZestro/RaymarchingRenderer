#ifndef RAYMARCHINGCUDA_PRIMITIVES_CUH
#define RAYMARCHINGCUDA_PRIMITIVES_CUH

#include "ray_maths.cuh"

class Primitive {
private:
    union {
        struct {
            ray::vec3 loc;
            ray::vec3 rot;
            ray::vec3 scale;
        };
        struct {
            float data[9];
        };
    };


public:
    __device__
    Primitive(const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale);

    __device__
    virtual float SDF(const ray::vec3& p)=0;

    __device__
    virtual float Normal(const ray::vec3& p);
    __device__
    virtual ~Primitive();
};

class Sphere : public Primitive {
    private:

    public:
    __device__
    Sphere(const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale, const float _radius);

    __device__
    float SDF(const ray::vec3& p) override;

    __device__
    ~Sphere() override;
};

class Cube : public Primitive {
public:
    __device__
    Cube();

    __device__
    float SDF(const ray::vec3& p) override;

    __device__
    ~Cube() override;
};

class Mandelbulb : public Primitive {
public:
    __device__
    Mandelbulb();

    __device__
    float SDF(const ray::vec3& p) override;

    __device__
    ~Mandelbulb() override;
};

#endif