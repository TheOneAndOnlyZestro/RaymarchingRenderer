#ifndef RAYMARCHINGCUDA_PRIMITIVES_CUH
#define RAYMARCHINGCUDA_PRIMITIVES_CUH

#include <memory>

#include "ray_maths.cuh"

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
class Primitive {
private:
    union {
        struct {
            float id;
            ray::vec3 loc;
            ray::vec3 rot;
            ray::vec3 scale;
        };
        struct {
            float data[10];
        };
    };

public:

    Primitive(float _id, const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale);

    //Accessor functions
    virtual ray::vec3 getLoc() const;

    virtual ray::vec3 getRot() const;

    virtual ray::vec3 getScale() const;

    virtual PrimitiveType getType() const=0;
    virtual bool isOperator() const=0;

    virtual void getData(float* out, size_t* size) const;

    virtual size_t getSize() const;
    
    virtual ray::vec3* getLocRef();
    
    virtual ray::vec3* getRotRef();
    
    virtual ray::vec3* getScaleRef();

    
    virtual void setLoc(const ray::vec3& loc);
    
    virtual void setRot(const ray::vec3& rot);
    
    virtual void setScale(const ray::vec3& scale);

    float getID() const;
    virtual ~Primitive();

};

class Sphere : public Primitive {
    private:
    float radius;
    public:
    
    Sphere(float id,const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale, const float _radius);

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
    static void SphereSDFF(const ray::vec3& p,const float* input,size_t* size, float* out);

    __device__ __host__
    static void SphereSDFFNorm(const ray::vec3& p,const float* input, ray::vec3* out);
    inline bool isOperator() const override {return false;}
    ~Sphere() override;
};

class Cube : public Primitive {
public:
    
    Cube(float id,const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale);

    virtual PrimitiveType getType() const override;

    __device__ __host__
    static void CubeSDF(const ray::vec3& p,
    const ray::vec3& loc, const ray::vec3& rot, const ray::vec3& scale,
    size_t* size, float* out);

    __device__ __host__
    static void CubeSDFF(const ray::vec3& p,const float* input, size_t* size, float* out);

    __device__ __host__
    static void CubeSDFFNorm(const ray::vec3& p,const float* input, ray::vec3* out);

    inline bool isOperator() const override {return false;}
    ~Cube() override;
};

class Mandelbulb : public Primitive {
private:
    unsigned int iterations;
    float exponent;
public:
    
    Mandelbulb(float id,const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale, const unsigned int _iterations, const float _exponent);

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
    static void MandelbulbSDFF(const ray::vec3& p,const float* input,size_t* size, float* out);

    __device__ __host__
    static void MandelbulbSDFFNorm(const ray::vec3& p,const float* input, ray::vec3* out);
    inline bool isOperator() const override {return false;}
    ~Mandelbulb() override;
};
class Line : public Primitive {
private:
    ray::vec3 a;
    ray::vec3 b;
    float radius;
public:
    Line(float id,const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale, const ray::vec3& _a, const ray::vec3& _b, const float _radius);

    void getData(float* out, size_t* size) const override;
    size_t getSize() const override;

    PrimitiveType getType() const override;

    float getRadius() const;
    void setRadius(const float _radius);

    ray::vec3 getA() const;
    ray::vec3 getB() const;

    ray::vec3* getARef();
    ray::vec3* getBRef();
    float* getRadiusRef();
    void setA(const ray::vec3& _a);
    void setB(const ray::vec3& _b);

    __device__ __host__
    static void LineSDF(const ray::vec3& p,const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale,
    const ray::vec3& _a, const ray::vec3& _b, const float _radius,size_t* size, float* out);

    __device__ __host__
    static void LineSDFF(const ray::vec3& p,const float* input,size_t* size, float* out);

    __device__ __host__
    static void LineSDFFNorm(const ray::vec3& p, const float* input, ray::vec3* out);
    inline bool isOperator() const override {return false;}
};
class BinaryOperator: public Primitive {
private:
    std::shared_ptr<Primitive> p1;
    std::shared_ptr<Primitive> p2;
public:

    BinaryOperator(float id,const std::shared_ptr<Primitive>& _p1,const std::shared_ptr<Primitive>& _p2);
    virtual PrimitiveType getType() const override;
    inline bool isOperator() const override {return true;}
    std::shared_ptr<Primitive> getP1() const;
    std::shared_ptr<Primitive> getP2() const;
};
class Union : public BinaryOperator {
    public:
    Union(float id,const std::shared_ptr<Primitive>& _p1,const std::shared_ptr<Primitive>& _p2);
    virtual PrimitiveType getType() const override;

    __device__ __host__
    static void UnionSDFF(const float d1, const float d2, size_t *size, float *out);

    __device__ __host__
    static void UnionSDFFNorm(float d1, float d2,const ray::vec3& n1, const ray::vec3&n2,ray::vec3* out,float* dout);
};

class Intersect : public BinaryOperator {
public:
    __device__ __host__
    Intersect(float id,const std::shared_ptr<Primitive>& _p1,const std::shared_ptr<Primitive>& _p2);
    virtual PrimitiveType getType() const override;

    __device__ __host__
    static void IntersectSDFF(const float d1, const float d2, size_t *size, float *out);

    __device__ __host__
    static void IntersectSDFFNorm(float d1, float d2,const ray::vec3& n1, const ray::vec3&n2,ray::vec3* out, float* dout);

};
#endif