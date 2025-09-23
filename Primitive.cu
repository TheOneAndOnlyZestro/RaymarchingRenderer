#include "Primitive.cuh"

//Parent Primitive
Primitive::Primitive(float _id, const ray::vec3 &_loc, const ray::vec3 &_rot, const ray::vec3 &_scale): loc(_loc), rot(_rot), scale(_scale), id(_id) {};

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
        *size = 10;
    if (out!=nullptr)
        memcpy_s(out,sizeof(float) * 10 ,data, sizeof(float) * 10);
}

size_t Primitive::getSize() const {
    return getPrimSize(PrimitiveType::PrimitiveType::CUBE);
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

float Primitive::getID() const {
    return id;
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

    //printf("LOC: (%f,%f,%f), ROT: (%f,%f,%f), SCALE:(%f,%f,%f) \n", input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7], input[8]);

    CubeSDF(p, &input[0], &input[3], &input[6], size,out);
}

void Cube::CubeSDFFNorm(const ray::vec3 &p, const float *input, ray::vec3 *out) {
    size_t size =0;
    float dxp,dxn,dyp,dyn,dzp,dzn;
    CubeSDF(p + ray::vec3(EPSILON,0.0f,0.0f),&input[0], &input[3], &input[6],&size,&dxp);
    CubeSDF(p - ray::vec3(EPSILON,0.0f,0.0f),&input[0], &input[3], &input[6],&size,&dxn);

    CubeSDF(p + ray::vec3(0.0f,EPSILON,0.0f),&input[0], &input[3], &input[6],&size,&dyp);
    CubeSDF(p - ray::vec3(0.0f,EPSILON,0.0f),&input[0], &input[3], &input[6], &size,&dyn);

    CubeSDF(p + ray::vec3(0.0f,0.0f,EPSILON),&input[0], &input[3], &input[6],&size,&dzp);
    CubeSDF(p - ray::vec3(0.0f,0.0f,EPSILON),&input[0], &input[3], &input[6],&size,&dzn);


    *out = ray::normalize(ray::vec3(dxp-dxn, dyp-dyn, dzp-dzn) );
}

__device__ __host__
void Sphere::SphereSDF(const ray::vec3 &p, const ray::vec3 &loc, const ray::vec3 &rot, const ray::vec3 &scale,
                       const float radius, size_t *size, float *out) {

    if (size != nullptr) *size = 1;
    out[0]=ray::length(p-loc) - radius;
    //printf("%f\n", out[0]);
}


__device__ __host__
void Sphere::SphereSDFF(const ray::vec3& p,const float *input, size_t *size, float *out) {

    //printf("LOC: (%f,%f,%f), ROT: (%f,%f,%f), SCALE:(%f,%f,%f), RAD: %f \n", input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7], input[8], input[9]);

    SphereSDF(p, &input[0], &input[3], &input[6], input[9], size, out);
}
__device__ __host__
void Sphere::SphereSDFFNorm(const ray::vec3 &p, const float *input, ray::vec3 *out) {
    size_t size =0;
    float dxp,dxn,dyp,dyn,dzp,dzn;
    SphereSDF(p + ray::vec3(EPSILON,0.0f,0.0f),&input[0], &input[3], &input[6], input[9],&size,&dxp);
    SphereSDF(p - ray::vec3(EPSILON,0.0f,0.0f),&input[0], &input[3], &input[6], input[9],&size,&dxn);

    SphereSDF(p + ray::vec3(0.0f,EPSILON,0.0f),&input[0], &input[3], &input[6], input[9],&size,&dyp);
    SphereSDF(p - ray::vec3(0.0f,EPSILON,0.0f),&input[0], &input[3], &input[6], input[9],&size,&dyn);

    SphereSDF(p + ray::vec3(0.0f,0.0f,EPSILON),&input[0], &input[3], &input[6], input[9],&size,&dzp);
    SphereSDF(p - ray::vec3(0.0f,0.0f,EPSILON),&input[0], &input[3], &input[6], input[9],&size,&dzn);


    *out = ray::normalize(ray::vec3(dxp-dxn, dyp-dyn, dzp-dzn) );

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
}

void Mandelbulb::MandelbulbSDFFNorm(const ray::vec3 &p, const float *input, ray::vec3 *out) {
    size_t size = 0;
    float dxp,dxn,dyp,dyn,dzp,dzn;
    MandelbulbSDF(p + ray::vec3(EPSILON,0.0f,0.0f), &input[0], &input[3], &input[6],(unsigned int)input[9], input[10], &size,&dxp);
    MandelbulbSDF(p - ray::vec3(EPSILON,0.0f,0.0f), &input[0], &input[3], &input[6],(unsigned int)input[9], input[10], &size,&dxn);

    MandelbulbSDF(p + ray::vec3(0.0f,EPSILON,0.0f), &input[0], &input[3], &input[6],(unsigned int)input[9], input[10], &size,&dyp);
    MandelbulbSDF(p - ray::vec3(0.0f,EPSILON,0.0f), &input[0], &input[3], &input[6],(unsigned int)input[9], input[10], &size,&dyn);

    MandelbulbSDF(p + ray::vec3(0.0f,0.0f,EPSILON), &input[0], &input[3], &input[6],(unsigned int)input[9], input[10], &size,&dzp);
    MandelbulbSDF(p - ray::vec3(0.0f,0.0f,EPSILON), &input[0], &input[3], &input[6],(unsigned int)input[9], input[10], &size,&dzn);


    *out = ray::normalize(ray::vec3(dxp-dxn, dyp-dyn, dzp-dzn) );

}

//Sphere SDF
Sphere::Sphere(float id,const ray::vec3 &_loc, const ray::vec3 &_rot, const ray::vec3 &_scale, const float _radius)
    :Primitive(id,_loc, _rot, _scale), radius(_radius) {}

PrimitiveType Sphere::getType() const {
    return PrimitiveType::SPHERE;
}

void Sphere::getData(float *out, size_t *size) const {
    Primitive::getData(out, size);
    out[*size] = radius;
    *size += 1;
}

size_t Sphere::getSize() const {
    return getPrimSize(PrimitiveType::SPHERE);
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

Cube::Cube(float id,const ray::vec3 &_loc, const ray::vec3 &_rot, const ray::vec3 &_scale)
:Primitive(id,_loc,_rot,_scale) {}

PrimitiveType Cube::getType() const {
    return PrimitiveType::CUBE;
}

Cube::~Cube() {
}

Mandelbulb::Mandelbulb(float id,const ray::vec3 &_loc, const ray::vec3 &_rot, const ray::vec3 &_scale,
    const unsigned int _iterations, const float _exponent)
        :Primitive(id,_loc,_rot,_scale), iterations(_iterations), exponent(_exponent){}

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
    return getPrimSize(PrimitiveType::MANDELBROT);
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

Line::Line(float _id,const ray::vec3 &_loc, const ray::vec3 &_rot, const ray::vec3 &_scale, const ray::vec3 &_a,
    const ray::vec3 &_b, const float _radius)
:Primitive(_id, _loc,_rot,_scale), a(_a), b(_b), radius(_radius)
{}

void Line::getData(float *out, size_t *size) const {
    Primitive::getData(out, size);
    memcpy_s(out + *size,sizeof(float) * 3 , a.v, sizeof(float) * 3);
    *size += 3;
    memcpy_s(out + *size,sizeof(float) * 3 , b.v, sizeof(float) * 3);
    *size +=3;
    out[*size] = (float)radius;
    *size += 1;
}

size_t Line::getSize() const {
    return getPrimSize(PrimitiveType::LINE);
}

PrimitiveType Line::getType() const {
    return PrimitiveType::LINE;
}

float Line::getRadius() const {
    return radius;
}

void Line::setRadius(const float _radius) {
    radius = _radius;
}

ray::vec3 Line::getA() const {
    return a;
}

ray::vec3 Line::getB() const {
    return b;
}

ray::vec3 * Line::getARef() {
    return &a;
}

ray::vec3 * Line::getBRef() {
    return &b;
}

float * Line::getRadiusRef() {
    return &radius;
}

void Line::setA(const ray::vec3 &_a) {
    a = _a;
}

void Line::setB(const ray::vec3 &_b) {
    b = _b;
}
__device__ __host__
void Line::LineSDF(const ray::vec3 &p, const ray::vec3 &_loc, const ray::vec3 &_rot, const ray::vec3 &_scale,
    const ray::vec3 &_a, const ray::vec3 &_b, const float _radius,size_t* size, float* out) {
    ray::vec3 pa = p - _a;
    ray::vec3 ba = _b - _a;
    float h = ray::clamp( ray::dot(pa,ba)/ray::dot(ba,ba), 0.0, 1.0 );
    out[0] = ray::length( pa - ba*h ) - _radius;
    *size=1;

}

void Line::LineSDFF(const ray::vec3 &p, const float *input, size_t *size, float *out) {
    LineSDF(p,&input[0],&input[3],&input[6],&input[9],&input[12],input[15],size,out);
}

void Line::LineSDFFNorm(const ray::vec3 &p, const float *input, ray::vec3 *out) {
    size_t size = 0;
    float dxp,dxn,dyp,dyn,dzp,dzn;
    LineSDF(p + ray::vec3(EPSILON,0.0f,0.0f),&input[0],&input[3],&input[6],&input[9],&input[12],input[13], &size,&dxp);
    LineSDF(p - ray::vec3(EPSILON,0.0f,0.0f),&input[0],&input[3],&input[6],&input[9],&input[12],input[13], &size,&dxn);

    LineSDF(p + ray::vec3(0.0f,EPSILON,0.0f),&input[0],&input[3],&input[6],&input[9],&input[12],input[13], &size,&dyp);
    LineSDF(p - ray::vec3(0.0f,EPSILON,0.0f),&input[0],&input[3],&input[6],&input[9],&input[12],input[13], &size,&dyn);

    LineSDF(p + ray::vec3(0.0f,0.0f,EPSILON),&input[0],&input[3],&input[6],&input[9],&input[12],input[13], &size,&dzp);
    LineSDF(p - ray::vec3(0.0f,0.0f,EPSILON),&input[0],&input[3],&input[6],&input[9],&input[12],input[13], &size,&dzn);

    *out = ray::normalize(ray::vec3(dxp-dxn, dyp-dyn, dzp-dzn) );
}

BinaryOperator::BinaryOperator(float id,const std::shared_ptr<Primitive> &_p1, const std::shared_ptr<Primitive> &_p2)
    : p1(_p1), p2(_p2),
Primitive(id, (_p1->getLoc() + _p2->getLoc())/2.f,
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

Union::Union(float id,const std::shared_ptr<Primitive> &_p1, const std::shared_ptr<Primitive> &_p2)
    :BinaryOperator(id,_p1,_p2){}

PrimitiveType Union::getType() const {
    return PrimitiveType::UNION;
}
__device__ __host__
void Union::UnionSDFF(const float d1, const float d2, size_t *size, float *out) {
    if (size != nullptr)
        *size = 4;
    out[0] = min(d1, d2);
}

void Union::UnionSDFFNorm(float d1, float d2, const ray::vec3 &n1, const ray::vec3 &n2,ray::vec3 *out, float* dout) {
    *dout = min(d1, d2);
    if (d1 < d2) {
        *out = n1;
    }else {
        *out = n2;
    }
}

Intersect::Intersect(float id,const std::shared_ptr<Primitive> &_p1, const std::shared_ptr<Primitive> &_p2)
:BinaryOperator(id,_p1,_p2) {}

PrimitiveType Intersect::getType() const {
    return PrimitiveType::INTERSECT;
}
__device__ __host__
void Intersect::IntersectSDFF(const float d1, const float d2, size_t *size, float *out) {
    if (size != nullptr)
        *size = 4;
    out[0] = max(d1, d2);
}
__device__ __host__
void Intersect::IntersectSDFFNorm(float d1, float d2,const ray::vec3& n1, const ray::vec3&n2, ray::vec3 *out, float* dout) {
    *dout = max(d1, d2);
    if (d1 > d2) {
        *out = n1;
    }else {
        *out = n2;
    }
}

