#ifndef RAYMARCHINGCUDA_PRIMITIVES_CUH
#define RAYMARCHINGCUDA_PRIMITIVES_CUH

#include <memory>
#include "PrimitiveUtils.cuh"
#include "ray_maths.cuh"

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

    virtual PrimitiveUtils::PrimitiveType getType() const=0;
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

    PrimitiveUtils::PrimitiveType getType() const override;

    //Accessor Functions
    void getData(float* out, size_t* size) const override;
    size_t getSize() const override;

    float getRadius() const;
    
    float* getRadiusRef();
    
    void setRadius(const float _radius);

    inline bool isOperator() const override {return false;}
    ~Sphere() override;
};

class Cube : public Primitive {
public:
    
    Cube(float id,const ray::vec3& _loc,const ray::vec3& _rot,const ray::vec3& _scale);

    PrimitiveUtils::PrimitiveType getType() const override;

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
    
    unsigned int getIterations() const;
    
    unsigned int* getIterationsRef();
    
    void setIterations(const unsigned int _iterations);

    void getData(float* out, size_t* size) const override;
    size_t getSize() const override;

    PrimitiveUtils::PrimitiveType getType() const override;

    
    float getExponent() const;
    
    float* getExponentRef();
    
    void setExponent(const float _exponent);

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

    PrimitiveUtils::PrimitiveType getType() const override;

    float getRadius() const;
    void setRadius(const float _radius);

    ray::vec3 getA() const;
    ray::vec3 getB() const;

    ray::vec3* getARef();
    ray::vec3* getBRef();
    float* getRadiusRef();
    void setA(const ray::vec3& _a);
    void setB(const ray::vec3& _b);

    inline bool isOperator() const override {return false;}
};
class BinaryOperator: public Primitive {
private:
    std::shared_ptr<Primitive> p1;
    std::shared_ptr<Primitive> p2;
public:

    BinaryOperator(float id,const std::shared_ptr<Primitive>& _p1,const std::shared_ptr<Primitive>& _p2);
    virtual PrimitiveUtils::PrimitiveType getType() const override;
    inline bool isOperator() const override {return true;}
    std::shared_ptr<Primitive> getP1() const;
    std::shared_ptr<Primitive> getP2() const;
};
class Union : public BinaryOperator {
    public:
    Union(float id,const std::shared_ptr<Primitive>& _p1,const std::shared_ptr<Primitive>& _p2);
    PrimitiveUtils::PrimitiveType getType() const override;

};

class Intersect : public BinaryOperator {
public:
    __device__ __host__
    Intersect(float id,const std::shared_ptr<Primitive>& _p1,const std::shared_ptr<Primitive>& _p2);
    PrimitiveUtils::PrimitiveType getType() const override;

};
#endif