#ifndef RAYMARCHINGCUDA_PRIMITIVES_CUH
#define RAYMARCHINGCUDA_PRIMITIVES_CUH

#include <memory>

#include "ray_maths.cuh"

enum class PrimitiveType {
    PRIMITIVE,
    SPHERE, CUBE, MANDELBROT, //Objects
    BINARY_OPERATOR,
    UNION, INTERSECT    //Operators
};

inline const char* DebugPrim(PrimitiveType p) {
    switch (p) {
        case PrimitiveType::PRIMITIVE:
            return "PrimitiveType::PRIMITIVE";
        case PrimitiveType::CUBE:
            return "PrimitiveType::CUBE";
        case PrimitiveType::SPHERE:
            return "PrimitiveType::SPHERE";
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

inline size_t getPrimSize(PrimitiveType p) {
    switch (p) {
        case PrimitiveType::PRIMITIVE:
            return 9;
        case PrimitiveType::CUBE:
            return 9;
        case PrimitiveType::SPHERE:
            return 10;
        case PrimitiveType::MANDELBROT:
            return 11;
        case PrimitiveType::BINARY_OPERATOR:
            return 9;
        case PrimitiveType::UNION:
            return 9;
        case PrimitiveType::INTERSECT:
            return 9;
        default:
            return -1;
    }
}
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

    __device__ __host__
    virtual void SDF(const float* input,size_t* size, float* out)=0;

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
    
    virtual ~Primitive();

};

class Sphere : public Primitive {
    private:
    float radius;
    public:
    
    Sphere(const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale, const float _radius);

    __host__ __device__
    void SDF(const float* input,size_t* size, float* out) override;

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
    static void SphereSDFF(const float* input,size_t* size, float* out);
    inline bool isOperator() const override {return false;}
    ~Sphere() override;
};

class Cube : public Primitive {
public:
    
    Cube(const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale);

    __host__ __device__
    void SDF(const float* input,size_t* size, float* out) override;
    
    virtual PrimitiveType getType() const override;

    __device__ __host__
    static void CubeSDF(const ray::vec3& p,
    const ray::vec3& loc, const ray::vec3& rot, const ray::vec3& scale,
    size_t* size, float* out);

    __device__ __host__
    static void CubeSDFF(const float* input, size_t* size, float* out);
    inline bool isOperator() const override {return false;}
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

    __host__ __device__
    void SDF(const float* input,size_t* size, float* out) override;

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
    static void MandelbulbSDFF(const float* input,size_t* size, float* out);
    inline bool isOperator() const override {return false;}
    ~Mandelbulb() override;
};

class BinaryOperator: public Primitive {
private:
    std::shared_ptr<Primitive> p1;
    std::shared_ptr<Primitive> p2;
public:

    BinaryOperator(const std::shared_ptr<Primitive>& _p1,const std::shared_ptr<Primitive>& _p2);
    virtual PrimitiveType getType() const override;
    inline bool isOperator() const override {return true;}
    std::shared_ptr<Primitive> getP1() const;
    std::shared_ptr<Primitive> getP2() const;
};
class Union : public BinaryOperator {
    public:
    Union(const std::shared_ptr<Primitive>& _p1,const std::shared_ptr<Primitive>& _p2);
    virtual PrimitiveType getType() const override;

    __host__ __device__
    void SDF(const float* input,size_t* size, float* out) override;
    __device__ __host__
    static void UnionSDFF(const float* input, size_t* size, float* out);

};

class Intersect : public BinaryOperator {
public:
    Intersect(const std::shared_ptr<Primitive>& _p1,const std::shared_ptr<Primitive>& _p2);
    virtual PrimitiveType getType() const override;

    __host__ __device__
    void SDF(const float* input,size_t* size, float* out) override;
    __device__ __host__
    static void IntersectSDFF(const float* input, size_t* size, float* out);

};
#endif