//
// Created by Andrew on 8/19/2025.
//

#ifndef RAYMARCHINGCUDA_RAY_MATHS_CUH
#define RAYMARCHINGCUDA_RAY_MATHS_CUH

#ifndef PI
#define PI acos(-1.0)
#endif
#ifndef EPSILON
#define EPSILON 1.e-6
#endif
//This section deals with vec3 structures and there corresponding functions
namespace ray {

    struct vec3 {
        union {
            struct {float x, y, z; };
            struct {float v[3]; };
            struct { float r, g, b; };
        };

        __device__ __host__
        vec3(float _x, float _y, float _z):x(_x), y(_y), z(_z){}
        vec3(const float* d):x(d[0]), y(d[1]), z(d[2]) {}
        __device__ __host__
        vec3(): x(0.f), y(0.f), z(0.f){}

        __device__ __host__
        inline explicit operator float*() {
            return v;
        }
    };

    __device__ __host__
    inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
        return vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
    }

    __device__ __host__
    inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
        return vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
    }

    __device__ __host__
    inline float length(const vec3& v1) {
        return sqrt(v1.x*v1.x + v1.y*v1.y + v1.z * v1.z);
    }

    __device__ __host__
    inline vec3 operator * (const vec3& v1, const float scalar) {
        return vec3(v1.x * scalar, v1.y * scalar, v1.z * scalar);
    }
    __device__ __host__
    inline vec3 operator * (const float scalar, const vec3& v1) {
        return vec3(v1.x * scalar, v1.y * scalar, v1.z * scalar);
    }

    __device__ __host__
    inline vec3 operator / (const vec3& v1, const float scalar) {
        return vec3(v1.x / scalar, v1.y / scalar, v1.z / scalar);
    }

    __device__ __host__
    inline vec3 normalize(const vec3& v1) {
        return v1 / length(v1);
    }

    __device__ __host__
    inline float dot(const vec3& v1, const vec3& v2) {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }

    __device__ __host__
    inline vec3 cross(const vec3& v1, const vec3& v2) {
        return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
    }

    __device__ __host__
    inline vec3 abs(const vec3& v1) {
        return vec3( ::fabsf(v1.x) , ::fabsf(v1.y), ::fabsf(v1.z));
    }

    __device__ __host__
    inline vec3 max(const vec3& v1, const float scalar) {
        return vec3(v1.x > scalar ? v1.x : scalar, v1.y > scalar? v1.y : scalar, v1.z > scalar? v1.z : scalar);
    }

    __device__ __host__
    inline vec3 min(const vec3& v1, const float scalar) {
        return vec3(v1.x < scalar ? v1.x : scalar, v1.y < scalar? v1.y : scalar, v1.z < scalar? v1.z : scalar);
    }

    __device__ __host__
    inline float compMax(const vec3& v1) {
        return ::fmaxf( ::fmaxf(v1.x, v1.y), v1.z );
    }

    __device__ __host__
    inline float compMin(const vec3& v1) {
        return ::fminf( ::fminf(v1.x, v1.y), v1.z );
    }

    __device__ __host__
    inline float clamp(float v, float a, float b) {
        return v > b ? b : (v < a ? a : v);
    }

    __device__  __host__ inline ray::vec3 fract(const ray::vec3& p) {
        return ray::vec3(p.x - floorf(p.x), p.y - floorf(p.y),p.z - floorf(p.z));
    }

    __device__ __host__ inline ray::vec3 mod(const ray::vec3& p, const float scalar) {
        return ray::vec3(
            p.x - scalar * floorf(p.x/scalar),
              p.y - scalar * floorf(p.y/scalar),
            p.z - scalar * floorf(p.z/scalar));
    }
    __device__ __host__ inline ray::vec3 operator+(const ray::vec3& p, const float scalar) {
        return ray::vec3(p.x + scalar, p.y + scalar, p.z + scalar);
    }

    __device__ __host__ inline ray::vec3 operator-(const ray::vec3& p, const float scalar) {
        return ray::vec3(p.x - scalar, p.y - scalar, p.z - scalar);
    }

    __device__ __host__ inline ray::vec3 operator*(const ray::vec3& v1, const ray::vec3& v2) {
        return ray::vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
    }

    __device__ __host__ inline ray::vec3 operator^(const ray::vec3& v1, const float power) {
        //get spherical representation
        float ro = length(v1);
        float theta = atan2f(v1.y, v1.x);
        float phi = acosf(v1.z / (ro+(float)EPSILON));

        ro = powf(ro, power);
        theta = theta * power;
        phi = phi * power;

        return ray::vec3(ro * sinf(phi) * cosf(theta), ro * sinf(phi) * sinf(theta), ro * cosf(phi));

    }

    __device__ __host__ inline ray::vec3 rotate(const ray::vec3& v,const int axis,const float angle) {
        float c = cosf(angle), s = sinf(angle);
        switch (axis) {
            case 0: // rotate around +X (affects y,z)
                return ray::vec3(
                    v.x,
                    c*v.y - s*v.z,
                    s*v.y + c*v.z
                );
            case 1: // rotate around +Y (affects x,z)  NOTE the sign pattern
                return ray::vec3(
                    c*v.x + s*v.z,
                    v.y,
                   -s*v.x + c*v.z
                );
            case 2: // rotate around +Z (affects x,y)
                return ray::vec3(
                    c*v.x - s*v.y,
                    s*v.x + c*v.y,
                    v.z
                );
            default:
                return ray::vec3();
        }
    }
}
#endif //RAYMARCHINGCUDA_RAY_MATHS_CUH