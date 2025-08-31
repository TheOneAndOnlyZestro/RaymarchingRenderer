#include "Primitive.cuh"

//Parent Primitive
Primitive::Primitive(const ray::vec3 &_loc, const ray::vec3 &_rot, const ray::vec3 &_scale): loc(_loc), rot(_rot), scale(_scale) {};

ray::vec3 Primitive::Normal(const ray::vec3 &p) {
    return ray::vec3(
    SDF(p + ray::vec3(EPSILON,0.0f,0.0f),nullptr)[0] - SDF(p - ray::vec3(EPSILON,0.0f,0.0f), nullptr)[0],
    SDF(p + ray::vec3(0.0f,EPSILON,0.0f),nullptr)[0] - SDF(p - ray::vec3(0.0f,EPSILON,0.0f), nullptr)[0],
    SDF(p + ray::vec3(0.0f,0.0f,EPSILON),nullptr)[0] - SDF(p - ray::vec3(0.0f,0.0f,EPSILON), nullptr)[0]);
}

ray::vec3 Primitive::getLoc() const {
    return loc;
}

ray::vec3 Primitive::getRot() const {
    return rot;
}

ray::vec3 Primitive::getScale() const {
    return scale;
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

float * Primitive::operator()(const ray::vec3 &p, size_t *size) {
    return SDF(p, size);
}

Primitive::~Primitive() {}

//Sphere SDF
Sphere::Sphere(const ray::vec3 &_loc, const ray::vec3 &_rot, const ray::vec3 &_scale, const float _radius)
    :Primitive(_loc, _rot, _scale), radius(_radius) {}

float *Sphere::SDF(const ray::vec3 &p, size_t *size) {
    *size =1;
    float* d = new float[*size];
    d[0]=ray::length(p-getLoc()) - radius;
    return d;
}

float Sphere::getRadius() const {
    return radius;
}

void Sphere::setRadius(const float _radius) {
    this->radius = _radius;
}

Sphere::~Sphere() {}

Cube::Cube(const ray::vec3 &_loc, const ray::vec3 &_rot, const ray::vec3 &_scale)
:Primitive(_loc,_rot,_scale) {}


float * Cube::SDF(const ray::vec3 &p, size_t *size) {
    *size = 1;
    float* d = new float[*size];
    ray::vec3 q(abs(p - getLoc()) - getScale());
    d[0] = ray::length(ray::max(q, 0.f)) + min( ray::compMax(q) , 0.0);
    return d;
}

Cube::~Cube() {
}

Mandelbulb::Mandelbulb(const ray::vec3 &_loc, const ray::vec3 &_rot, const ray::vec3 &_scale,
    const unsigned int _iterations, const float _exponent)
        :Primitive(_loc,_rot,_scale), iterations(_iterations), exponent(_exponent){}

float * Mandelbulb::SDF(const ray::vec3 &p, size_t *size) {
    ray::vec3 zold(0.f,0.f,0.f);
    ray::vec3 znew(0.f,0.f,0.f);

    float dr = 1.0f;
    for (unsigned int i = 0; i < iterations; i++) {

        if (ray::length(zold) > 8.f) {
            break;
        }
        znew = (zold ^ exponent) + p;
        dr = (exponent * powf(length(zold), exponent-1.f) * dr) + 1.f;
        zold = znew;
    }

    *size = 2;
    float* d = new float[*size];
    d[0] = 0.5f * (ray::length(znew) * logf(ray::length(znew)) )/(dr+EPSILON);
    d[1] =  length(zold) - floorf(length(zold));
    return d;
}

unsigned int Mandelbulb::getIterations() const {
    return iterations;
}

void Mandelbulb::setIterations(const unsigned int _iterations) {
    this->iterations = _iterations;
}

float Mandelbulb::getExponent() const {
    return exponent;
}

void Mandelbulb::setExponent(const float _exponent) {
    this->exponent = _exponent;
}

Mandelbulb::~Mandelbulb() {}

