#include "Primitive.h"

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
    return getPrimSize(PrimitiveUtils::PrimitiveType::CUBE);
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
Sphere::Sphere(float id,const ray::vec3 &_loc, const ray::vec3 &_rot, const ray::vec3 &_scale, const float _radius)
    :Primitive(id,_loc, _rot, _scale), radius(_radius) {}

PrimitiveUtils::PrimitiveType Sphere::getType() const {
    return PrimitiveUtils::PrimitiveType::SPHERE;
}

void Sphere::getData(float *out, size_t *size) const {
    Primitive::getData(out, size);
    out[*size] = radius;
    *size += 1;
}

size_t Sphere::getSize() const {
    return getPrimSize(PrimitiveUtils::PrimitiveType::SPHERE);
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

PrimitiveUtils::PrimitiveType Cube::getType() const {
    return PrimitiveUtils::PrimitiveType::CUBE;
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
    return getPrimSize(PrimitiveUtils::PrimitiveType::MANDELBROT);
}

PrimitiveUtils::PrimitiveType Mandelbulb::getType() const {
    return PrimitiveUtils::PrimitiveType::MANDELBROT;
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
    return getPrimSize(PrimitiveUtils::PrimitiveType::LINE);
}

PrimitiveUtils::PrimitiveType Line::getType() const {
    return PrimitiveUtils::PrimitiveType::LINE;
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

BinaryOperator::BinaryOperator(float id,const std::shared_ptr<Primitive> &_p1, const std::shared_ptr<Primitive> &_p2)
    : p1(_p1), p2(_p2),
Primitive(id, (_p1->getLoc() + _p2->getLoc())/2.f,
    ray::vec3(),ray::vec3(1.f,1.f,1.f)) {}

PrimitiveUtils::PrimitiveType BinaryOperator::getType() const {
    return PrimitiveUtils::PrimitiveType::BINARY_OPERATOR;
}

std::shared_ptr<Primitive> BinaryOperator::getP1() const {
    return p1;
}

std::shared_ptr<Primitive> BinaryOperator::getP2() const {
    return p2;
}

Union::Union(float id,const std::shared_ptr<Primitive> &_p1, const std::shared_ptr<Primitive> &_p2)
    :BinaryOperator(id,_p1,_p2){}

PrimitiveUtils::PrimitiveType Union::getType() const {
    return PrimitiveUtils::PrimitiveType::UNION;
}

Intersect::Intersect(float id,const std::shared_ptr<Primitive> &_p1, const std::shared_ptr<Primitive> &_p2)
:BinaryOperator(id,_p1,_p2) {}

PrimitiveUtils::PrimitiveType Intersect::getType() const {
    return PrimitiveUtils::PrimitiveType::INTERSECT;
}

