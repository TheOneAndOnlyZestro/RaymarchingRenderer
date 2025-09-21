#include "Primitive.cuh"

//Parent Primitive
Primitive::Primitive(const ray::vec3 &_loc, const ray::vec3 &_rot, const ray::vec3 &_scale): loc(_loc), rot(_rot), scale(_scale)  {};

// ray::vec3 Primitive::Normal(const ray::vec3 &p) const{
//     float dxp,dxn,dyp,dyn,dzp,dzn;
//     SDF(p + ray::vec3(EPSILON,0.0f,0.0f),nullptr,&dxp);
//     SDF(p - ray::vec3(EPSILON,0.0f,0.0f),nullptr,&dxn);
//
//     SDF(p + ray::vec3(0.0f,EPSILON,0.0f),nullptr,&dyp);
//     SDF(p - ray::vec3(0.0f,EPSILON,0.0f),nullptr,&dyn);
//
//     SDF(p + ray::vec3(0.0f,0.0f,EPSILON),nullptr,&dzp);
//     SDF(p - ray::vec3(0.0f,0.0f,EPSILON),nullptr,&dzn);
//
//
//     return ray::normalize(ray::vec3(dxp-dxn, dyp-dyn, dzp-dzn) );
//     }

ray::vec3 Primitive::getLoc() const {
    return loc;
}

ray::vec3 Primitive::getRot() const {
    return rot;
}

ray::vec3 Primitive::getScale() const {
    return scale;
}


void Primitive::getData(float *out, size_t *size) const{
    if (size!=nullptr)
        *size = 9;
    if (out!=nullptr)
        memcpy_s(out,sizeof(float) * 9 ,data, sizeof(float) * 9);
}

size_t Primitive::getSize() const {
    return 9;
}

ray::vec3 * Primitive::getLocRef() {
    return &loc;
}

ray::vec3 * Primitive::getRotRef() {
    return &rot;
}

ray::vec3 * Primitive::getScaleRef() {
    return &scale;
}

void Primitive::setLoc(const ray::vec3 &loc) {
    this->loc = loc;
}

void Primitive::setRot(const ray::vec3 &rot) {
    this->rot = rot;
}

void Primitive::setScale(const ray::vec3 &scale) {
    this->scale = scale;
}

Primitive::~Primitive() {}

__device__ __host__
void Cube::CubeSDF(const ray::vec3 &p, const ray::vec3 &loc, const ray::vec3 &rot, const ray::vec3 &scale,
    size_t *size, float *out) {

    if (size != nullptr) *size = 1;
    ray::vec3 q(abs(p - loc) - scale);
    out[0] = ray::length(ray::max(q, 0.f)) + min( ray::compMax(q) , 0.0);
}
__device__ __host__
void Cube::CubeSDFF(const ray::vec3& p, const float *input, size_t *size, float *out) {
    CubeSDF(p, &input[0], &input[3], &input[6], size,out);


    float dxp,dxn,dyp,dyn,dzp,dzn;
    CubeSDF(p + ray::vec3(EPSILON,0.0f,0.0f),&input[0], &input[3], &input[6],size,&dxp);
    CubeSDF(p - ray::vec3(EPSILON,0.0f,0.0f),&input[0], &input[3], &input[6],size,&dxn);

    CubeSDF(p + ray::vec3(0.0f,EPSILON,0.0f),&input[0], &input[3], &input[6],size,&dyp);
    CubeSDF(p - ray::vec3(0.0f,EPSILON,0.0f),&input[0], &input[3], &input[6],size,&dyn);

    CubeSDF(p + ray::vec3(0.0f,0.0f,EPSILON),&input[0], &input[3], &input[6],size,&dzp);
    CubeSDF(p - ray::vec3(0.0f,0.0f,EPSILON),&input[0], &input[3], &input[6],size,&dzn);


    ray::vec3 norm = ray::normalize(ray::vec3(dxp-dxn, dyp-dyn, dzp-dzn) );
    out[1] = norm.x;
    out[2] = norm.y;
    out[3] = norm.z;

}
__device__ __host__
void Sphere::SphereSDF(const ray::vec3 &p, const ray::vec3 &loc, const ray::vec3 &rot, const ray::vec3 &scale,
                       const float radius, size_t *size, float *out) {
    if (size != nullptr) *size = 1;
    out[0]=ray::length(p-loc) - radius;
}


__device__ __host__
void Sphere::SphereSDFF(const ray::vec3& p,const float *input, size_t *size, float *out) {
    SphereSDF(p, &input[0], &input[3], &input[6], input[9], size, out);

    float dxp,dxn,dyp,dyn,dzp,dzn;
    SphereSDF(p + ray::vec3(EPSILON,0.0f,0.0f),&input[0], &input[3], &input[6], input[9],size,&dxp);
    SphereSDF(p - ray::vec3(EPSILON,0.0f,0.0f),&input[0], &input[3], &input[6], input[9],size,&dxn);

    SphereSDF(p + ray::vec3(0.0f,EPSILON,0.0f),&input[0], &input[3], &input[6], input[9],size,&dyp);
    SphereSDF(p - ray::vec3(0.0f,EPSILON,0.0f),&input[0], &input[3], &input[6], input[9],size,&dyn);

    SphereSDF(p + ray::vec3(0.0f,0.0f,EPSILON),&input[0], &input[3], &input[6], input[9],size,&dzp);
    SphereSDF(p - ray::vec3(0.0f,0.0f,EPSILON),&input[0], &input[3], &input[6], input[9],size,&dzn);


    ray::vec3 norm = ray::normalize(ray::vec3(dxp-dxn, dyp-dyn, dzp-dzn) );
    out[1] = norm.x;
    out[2] = norm.y;
    out[3] = norm.z;
}
__device__ __host__
void Mandelbulb::MandelbulbSDF(const ray::vec3 &p, const ray::vec3 &loc, const ray::vec3 &rot, const ray::vec3 &scale,
                               const unsigned int iterations, const float exponent, size_t *size, float *out) {
    ray::vec3 pnew =
        ray::rotate(ray::rotate(ray::rotate( (p - loc),0,rot.x * (PI/180.f)),1,rot.y * (PI/180.f)),2,rot.z * (PI/180.f));
    ray::vec3 zold(0.f,0.f,0.f);
    ray::vec3 znew(0.f,0.f,0.f);

    float dr = 1.0f;
    for (unsigned int i = 0; i < iterations; i++) {

        if (ray::length(zold) > 8.f) {
            break;
        }
        znew = (zold ^ exponent) + pnew;
        dr = (exponent * powf(length(zold), exponent-1.f) * dr) + 1.f;
        zold = znew;
    }

    if (size!=nullptr)
        *size = 2;
    out[0] = 0.5f * (ray::length(znew) * logf(ray::length(znew)) )/(dr+EPSILON);
    //float v = length(zold) - floorf(length(zold));
    //out[1] =  v;
}
__device__ __host__
void Mandelbulb::MandelbulbSDFF(const ray::vec3& p,const float *input, size_t *size, float *out) {
    MandelbulbSDF(p, &input[0], &input[3], &input[6],(unsigned int)input[9], input[10], size, out);

    float dxp,dxn,dyp,dyn,dzp,dzn;
    MandelbulbSDF(p + ray::vec3(EPSILON,0.0f,0.0f), &input[0], &input[3], &input[6],(unsigned int)input[9], input[10], size,&dxp);
    MandelbulbSDF(p - ray::vec3(EPSILON,0.0f,0.0f), &input[0], &input[3], &input[6],(unsigned int)input[9], input[10], size,&dxn);

    MandelbulbSDF(p + ray::vec3(0.0f,EPSILON,0.0f), &input[0], &input[3], &input[6],(unsigned int)input[9], input[10], size,&dyp);
    MandelbulbSDF(p - ray::vec3(0.0f,EPSILON,0.0f), &input[0], &input[3], &input[6],(unsigned int)input[9], input[10], size,&dyn);

    MandelbulbSDF(p + ray::vec3(0.0f,0.0f,EPSILON), &input[0], &input[3], &input[6],(unsigned int)input[9], input[10], size,&dzp);
    MandelbulbSDF(p - ray::vec3(0.0f,0.0f,EPSILON), &input[0], &input[3], &input[6],(unsigned int)input[9], input[10], size,&dzn);


    ray::vec3 norm = ray::normalize(ray::vec3(dxp-dxn, dyp-dyn, dzp-dzn) );
    out[1] = norm.x;
    out[2] = norm.y;
    out[3] = norm.z;
}

//Sphere SDF
Sphere::Sphere(const ray::vec3 &_loc, const ray::vec3 &_rot, const ray::vec3 &_scale, const float _radius)
    :Primitive(_loc, _rot, _scale), radius(_radius) {}

PrimitiveType Sphere::getType() const {
    return PrimitiveType::SPHERE;
}

void Sphere::getData(float *out, size_t *size) const {
    Primitive::getData(out, size);
    out[*size] = radius;
    *size += 1;
}

size_t Sphere::getSize() const {
    return Primitive::getSize() + 1;
}


float Sphere::getRadius() const {
    return radius;
}

float * Sphere::getRadiusRef() {
    return &radius;
}

void Sphere::setRadius(const float _radius) {
    this->radius = _radius;
}

Sphere::~Sphere() {}

Cube::Cube(const ray::vec3 &_loc, const ray::vec3 &_rot, const ray::vec3 &_scale)
:Primitive(_loc,_rot,_scale) {}

PrimitiveType Cube::getType() const {
    return PrimitiveType::CUBE;
}

Cube::~Cube() {
}

Mandelbulb::Mandelbulb(const ray::vec3 &_loc, const ray::vec3 &_rot, const ray::vec3 &_scale,
    const unsigned int _iterations, const float _exponent)
        :Primitive(_loc,_rot,_scale), iterations(_iterations), exponent(_exponent){}

unsigned int Mandelbulb::getIterations() const {
    return iterations;
}

unsigned int * Mandelbulb::getIterationsRef() {
    return &iterations;
}

void Mandelbulb::setIterations(const unsigned int _iterations) {
    this->iterations = _iterations;
}

void Mandelbulb::getData(float *out, size_t *size) const {
    Primitive::getData(out, size);
    out[*size] = (float)iterations;
    out[*size +1] = exponent;
    *size += 2;
}

size_t Mandelbulb::getSize() const {
    return Primitive::getSize() + 2;
}

PrimitiveType Mandelbulb::getType() const {
    return PrimitiveType::MANDELBROT;
}

float Mandelbulb::getExponent() const {
    return exponent;
}

float * Mandelbulb::getExponentRef() {
    return &exponent;
}

void Mandelbulb::setExponent(const float _exponent) {
    this->exponent = _exponent;
}

Mandelbulb::~Mandelbulb() {}

BinaryOperator::BinaryOperator(const std::shared_ptr<Primitive> &_p1, const std::shared_ptr<Primitive> &_p2)
    : p1(_p1), p2(_p2),
Primitive( (_p1->getLoc() + _p2->getLoc())/2.f,
    ray::vec3(),ray::vec3(1.f,1.f,1.f)) {}

PrimitiveType BinaryOperator::getType() const {
    return PrimitiveType::BINARY_OPERATOR;
}

std::shared_ptr<Primitive> BinaryOperator::getP1() const {
    return p1;
}

std::shared_ptr<Primitive> BinaryOperator::getP2() const {
    return p2;
}

Union::Union(const std::shared_ptr<Primitive> &_p1, const std::shared_ptr<Primitive> &_p2)
    :BinaryOperator(_p1,_p2){}

PrimitiveType Union::getType() const {
    return PrimitiveType::UNION;
}
__device__ __host__
void Union::UnionSDFF(const float d1, const float d2, const ray::vec3& n1, const ray::vec3& n2, size_t *size, float *out) {
    if (size != nullptr)
        *size = 4;
    out[0] = min(d1, d2);
    out[1] = min(n1.x, n2.x);
    out[2] = min(n1.y, n2.y);
    out[3] = min(n1.z, n2.z);

}
Intersect::Intersect(const std::shared_ptr<Primitive> &_p1, const std::shared_ptr<Primitive> &_p2)
:BinaryOperator(_p1,_p2) {}

PrimitiveType Intersect::getType() const {
    return PrimitiveType::INTERSECT;
}
__device__ __host__
void Intersect::IntersectSDFF(const float d1, const float d2, const ray::vec3& n1, const ray::vec3& n2, size_t *size, float *out) {
    if (size != nullptr)
        *size = 4;
    out[0] = max(d1, -d2);
    out[1] = max(n1.x, -n2.x);
    out[2] = max(n1.y, -n2.y);
    out[3] = max(n1.z, -n2.z);
}

