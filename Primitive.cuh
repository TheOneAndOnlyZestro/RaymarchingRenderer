#ifndef RAYMARCHINGCUDA_PRIMITIVES_CUH
#define RAYMARCHINGCUDA_PRIMITIVES_CUH

#include <memory>

#include "ray_maths.cuh"

enum class PrimitiveType {
    PRIMITIVE,
    SPHERE, CUBE, MANDELBROT, //Objects
    UNION, INTERSECT    //Operators
};


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
    
    Primitive(const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale);
    //Accessor functions
    
    virtual ray::vec3 getLoc() const;
    
    virtual ray::vec3 getRot() const;
    
    virtual ray::vec3 getScale() const;
    
    virtual PrimitiveType getType() const=0;
    virtual bool isOperator() const;

    virtual void getData(float* out, size_t* size) const;

    virtual size_t getSize() const;
    
    virtual ray::vec3* getLocRef();
    
    virtual ray::vec3* getRotRef();
    
    virtual ray::vec3* getScaleRef();

    
    virtual void setLoc(const ray::vec3& loc);
    
    virtual void setRot(const ray::vec3& rot);
    
    virtual void setScale(const ray::vec3& scale);
    
    virtual ~Primitive();

};

class Sphere : public Primitive {
    private:
    float radius;
    public:
    
    Sphere(const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale, const float _radius);

    
    virtual PrimitiveType getType() const override;

    //Accessor Functions
    virtual void getData(float* out, size_t* size) const override;
    virtual size_t getSize() const override;

    virtual float getRadius() const;
    
    virtual float* getRadiusRef();
    
    virtual void setRadius(const float _radius);

    __device__ __host__
    static void SphereSDF(const ray::vec3& p,
    const ray::vec3& loc, const ray::vec3& rot, const ray::vec3& scale, const float radius,
    size_t* size, float* out);

    __device__ __host__
    static void SphereSDF(const float* input,size_t* size, float* out);
    ~Sphere() override;
};

class Cube : public Primitive {
public:
    
    Cube(const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale);

    
    virtual PrimitiveType getType() const override;

    __device__ __host__
    static void CubeSDF(const ray::vec3& p,
    const ray::vec3& loc, const ray::vec3& rot, const ray::vec3& scale,
    size_t* size, float* out);

    __device__ __host__
    static void CubeSDF(const float* input, size_t* size, float* out);
    ~Cube() override;
};

class Mandelbulb : public Primitive {
private:
    unsigned int iterations;
    float exponent;
public:
    
    Mandelbulb(const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale, const unsigned int _iterations, const float _exponent);

    //Accessor Functions
    
    virtual unsigned int getIterations() const;
    
    virtual unsigned int* getIterationsRef();
    
    virtual void setIterations(const unsigned int _iterations);


    virtual void getData(float* out, size_t* size) const override;
    virtual size_t getSize() const override;

    virtual PrimitiveType getType() const override;

    
    virtual float getExponent() const;
    
    virtual float* getExponentRef();
    
    virtual void setExponent(const float _exponent);

    __device__ __host__
    static void MandelbulbSDF(const ray::vec3& p,
    const ray::vec3& loc, const ray::vec3& rot, const ray::vec3& scale, const unsigned int iterations, const float exponent,
    size_t* size, float* out);

    __device__ __host__
    static void MandelbulbSDF(const float* input,size_t* size, float* out);
    ~Mandelbulb() override;
};

class Union : public Primitive {
    private:
    std::shared_ptr<Primitive> p1;
    std::shared_ptr<Primitive> p2;
    public:

    Union(const std::shared_ptr<Primitive>& _p1,const std::shared_ptr<Primitive>& _p2);
    virtual PrimitiveType getType() const override;

    std::shared_ptr<Primitive> getP1() const;
    std::shared_ptr<Primitive> getP2() const;
    __device__ __host__
    static void UnionSDF(const float* input, size_t* size, float* out);

};

class Intersect : public Primitive {
private:
    std::shared_ptr<Primitive> p1;
    std::shared_ptr<Primitive> p2;
public:
    Intersect(const std::shared_ptr<Primitive>& _p1,const std::shared_ptr<Primitive>& _p2);
    virtual PrimitiveType getType() const override;

    std::shared_ptr<Primitive> getP1() const;
    std::shared_ptr<Primitive> getP2() const;

    __device__ __host__
    static void IntersectSDF(const float* input, size_t* size, float* out);

};
#endif