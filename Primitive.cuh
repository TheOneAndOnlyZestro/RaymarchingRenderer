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
    virtual float* SDF(const ray::vec3& p, size_t* size)=0;
    __device__
    virtual ray::vec3 Normal(const ray::vec3& p);

    //Accessor functions
    __device__
    virtual ray::vec3 getLoc() const;
    __device__
    virtual ray::vec3 getRot() const;
    __device__
    virtual ray::vec3 getScale() const;

    __device__
    virtual void setLoc(const ray::vec3& loc);
    __device__
    virtual void setRot(const ray::vec3& rot);
    __device__
    virtual void setScale(const ray::vec3& scale);

    __device__
    virtual float* operator()(const ray::vec3& p, size_t* size);

    __device__
    virtual ~Primitive();
};

class Sphere : public Primitive {
    private:
    float radius;
    public:
    __device__
    Sphere(const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale, const float _radius);

    __device__
    float* SDF(const ray::vec3& p, size_t* size) override;

    //Accessor Functions
    __device__
    virtual float getRadius() const;
    __device__
    virtual void setRadius(const float _radius);

    __device__
    ~Sphere() override;
};

class Cube : public Primitive {
public:
    __device__
    Cube(const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale);

    __device__
    float* SDF(const ray::vec3& p, size_t* size) override;

    __device__
    ~Cube() override;
};

class Mandelbulb : public Primitive {
private:
    unsigned int iterations;
    float exponent;
public:
    __device__
    Mandelbulb(const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale, const unsigned int _iterations, const float _exponent);

    __device__
    float* SDF(const ray::vec3& p, size_t* size) override;

    //Accessor Functions
    __device__
    virtual unsigned int getIterations() const;
    __device__
    virtual void setIterations(const unsigned int _iterations);

    __device__
    virtual float getExponent() const;
    __device__
    virtual void setExponent(const float _exponent);


    __device__
    ~Mandelbulb() override;
};



#endif