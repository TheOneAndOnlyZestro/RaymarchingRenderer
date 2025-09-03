#ifndef RAYMARCHINGCUDA_PRIMITIVES_CUH
#define RAYMARCHINGCUDA_PRIMITIVES_CUH

#include "ray_maths.cuh"

enum class PrimitiveType { PRIMITIVE, SPHERE, CUBE, MANDELBROT };


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
    __device__ __host__
    Primitive(const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale);
    //Accessor functions
    __device__ __host__
    virtual ray::vec3 getLoc() const;
    __device__ __host__
    virtual ray::vec3 getRot() const;
    __device__ __host__
    virtual ray::vec3 getScale() const;
    __device__ __host__
    virtual PrimitiveType getType() const=0;

    __device__ __host__
    virtual ray::vec3* getLocRef();
    __device__ __host__
    virtual ray::vec3* getRotRef();
    __device__ __host__
    virtual ray::vec3* getScaleRef();

    __device__ __host__
    virtual void setLoc(const ray::vec3& loc);
    __device__ __host__
    virtual void setRot(const ray::vec3& rot);
    __device__ __host__
    virtual void setScale(const ray::vec3& scale);
    __device__ __host__
    virtual ~Primitive();






};

class Sphere : public Primitive {
    private:
    float radius;
    public:
    __device__ __host__
    Sphere(const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale, const float _radius);

    __device__ __host__
    virtual PrimitiveType getType() const override;

    //Accessor Functions
    __device__ __host__
    virtual float getRadius() const;
    __device__ __host__
    virtual float* getRadiusRef();
    __device__ __host__
    virtual void setRadius(const float _radius);

    __device__ __host__
    static void SphereSDF(const ray::vec3& p,
    const ray::vec3& loc, const ray::vec3& rot, const ray::vec3& scale, const float radius,
    size_t* size, float* out);

    __device__ __host__
    ~Sphere() override;
};

class Cube : public Primitive {
public:
    __device__ __host__
    Cube(const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale);

    __device__ __host__
    virtual PrimitiveType getType() const override;

    __device__ __host__
    static void CubeSDF(const ray::vec3& p,
    const ray::vec3& loc, const ray::vec3& rot, const ray::vec3& scale,
    size_t* size, float* out);

    __device__ __host__
    ~Cube() override;
};

class Mandelbulb : public Primitive {
private:
    unsigned int iterations;
    float exponent;
public:
    __device__ __host__
    Mandelbulb(const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale, const unsigned int _iterations, const float _exponent);

    //Accessor Functions
    __device__ __host__
    virtual unsigned int getIterations() const;
    __device__ __host__
    virtual unsigned int* getIterationsRef();
    __device__ __host__
    virtual void setIterations(const unsigned int _iterations);

    __device__ __host__
    virtual PrimitiveType getType() const override;

    __device__ __host__
    virtual float getExponent() const;
    __device__ __host__
    virtual float* getExponentRef();
    __device__ __host__
    virtual void setExponent(const float _exponent);

    __device__ __host__
    static void MandelbulbSDF(const ray::vec3& p,
    const ray::vec3& loc, const ray::vec3& rot, const ray::vec3& scale, const unsigned int iterations, const float exponent,
    size_t* size, float* out);

    __device__ __host__
    ~Mandelbulb() override;
};



#endif