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
    struct vec3;
    struct vec4;

    struct vec4 {
    union {
        struct { float x, y, z, w; };
        struct { float r, g, b, a; };
        float v[4];
    };

    __device__ __host__
    vec4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}

    __device__ __host__
    vec4(const float* d) : x(d[0]), y(d[1]), z(d[2]), w(d[3]) {}

    __device__ __host__
    vec4() : x(0.f), y(0.f), z(0.f), w(0.f) {}

    __device__ __host__
    inline explicit operator float*() { return v; }
};


__device__ __host__
inline vec4 operator+(const vec4& v1, const vec4& v2) {
    return vec4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
}

__device__ __host__
inline vec4 operator-(const vec4& v1, const vec4& v2) {
    return vec4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
}

__device__ __host__
inline vec4 operator*(const vec4& v1, float scalar) {
    return vec4(v1.x * scalar, v1.y * scalar, v1.z * scalar, v1.w * scalar);
}

__device__ __host__
inline vec4 operator*(float scalar, const vec4& v1) {
    return v1 * scalar;
}

__device__ __host__
inline vec4 operator/(const vec4& v1, float scalar) {
    return vec4(v1.x / scalar, v1.y / scalar, v1.z / scalar, v1.w / scalar);
}

__device__ __host__
inline float dot(const vec4& v1, const vec4& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

__device__ __host__
inline float length(const vec4& v1) {
    return sqrtf(dot(v1, v1));
}

__device__ __host__
inline vec4 normalize(const vec4& v1) {
    return v1 / length(v1);
}

__device__ __host__
inline vec4 abs(const vec4& v1) {
    return vec4(::fabsf(v1.x), ::fabsf(v1.y), ::fabsf(v1.z), ::fabsf(v1.w));
}

    __device__ __host__
inline vec4 fract(const vec4& p) {
    return vec4(
        p.x - floorf(p.x),
        p.y - floorf(p.y),
        p.z - floorf(p.z),
        p.w - floorf(p.w)
    );
}

    __device__ __host__
    inline vec4 mod(const vec4& p, float scalar) {
    return vec4(
        p.x - scalar * floorf(p.x / scalar),
        p.y - scalar * floorf(p.y / scalar),
        p.z - scalar * floorf(p.z / scalar),
        p.w - scalar * floorf(p.w / scalar)
    );
}

__device__ __host__
inline vec4 max(const vec4& v1, float scalar) {
    return vec4(
        v1.x > scalar ? v1.x : scalar,
        v1.y > scalar ? v1.y : scalar,
        v1.z > scalar ? v1.z : scalar,
        v1.w > scalar ? v1.w : scalar
    );
}

__device__ __host__
inline vec4 min(const vec4& v1, float scalar) {
    return vec4(
        v1.x < scalar ? v1.x : scalar,
        v1.y < scalar ? v1.y : scalar,
        v1.z < scalar ? v1.z : scalar,
        v1.w < scalar ? v1.w : scalar
    );
}

__device__ __host__
inline float compMax(const vec4& v1) {
    return fmaxf(fmaxf(v1.x, v1.y), fmaxf(v1.z, v1.w));
}

__device__ __host__
inline float compMin(const vec4& v1) {
    return fminf(fminf(v1.x, v1.y), fminf(v1.z, v1.w));
}

struct vec3 {
        union {
            struct {float x, y, z; };
            struct {float v[3]; };
            struct { float r, g, b; };
        };

        __device__ __host__
        vec3(float _x, float _y, float _z):x(_x), y(_y), z(_z){}
        __device__ __host__
        vec3(const float* d):x(d[0]), y(d[1]), z(d[2]) {}
        __device__ __host__
        vec3(): x(0.f), y(0.f), z(0.f){}

        __device__ __host__
        inline explicit operator float*() {
            return v;
        }

        __device__ __host__
        inline explicit operator vec4() {
            return vec4(x,y,z,1.f);
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


struct mat4x4 {
        union {
            struct {
                float m[4][4];
            };
            struct {
                float v[16];
            };
            struct {
                ray::vec4 vc1, vc2, vc3, vc4;
            };
        };

        __device__ __host__
        inline mat4x4(const float* d) {
            memcpy_s(v, sizeof(float) * 16, d, sizeof(float) * 16);
        }
        inline mat4x4(const ray::vec4& v1, const ray::vec4& v2,const ray::vec4& v3, const vec4& v4, bool isRow = true)
         {
            if (isRow) {
                vc1 = v1;vc2 = v2;vc3=v3;
            }else {
                vc1 = ray::vec4(v1.x, v2.x, v3.x, v4.w);
                vc2 = ray::vec4(v1.y, v2.y, v3.y, v4.w);
                vc3 = ray::vec4(v1.z, v2.z, v3.z, v4.w);
            }

        }
        __device__ __host__
        inline mat4x4() {
            m[0][0] = 1.f;
            m[1][1] = 1.f;
            m[2][2] = 1.f;
            m[3][3] = 1.f;
        }

        __device__ __host__
        inline ray::vec4 getRow(unsigned int index) const {
            return m[index];
        }

        __device__ __host__
        inline ray::vec4 getCol(unsigned int index) const {
            return {m[0][index], m[1][index], m[2][index], m[3][index]};
        }

    };

    __device__ __host__
    inline mat4x4 operator+(const mat4x4& m1, const mat4x4& m2) {
        return {m1.vc1 + m2.vc1, m1.vc2 + m2.vc2, m1.vc3 + m2.vc3, m1.vc4 + m2.vc4};
    }

    __device__ __host__
    inline mat4x4 operator-(const mat4x4& m1, const mat4x4& m2) {
        return {m1.vc1 - m2.vc1, m1.vc2 - m2.vc2, m1.vc3 - m2.vc3, m1.vc4 - m2.vc4};
    }

    __device__ __host__
    inline mat4x4 dot(const mat4x4& m1, const mat4x4& m2) {
        return { ray::vec4(dot(m1.getRow(0), m2.getCol(0)), dot(m1.getRow(0), m2.getCol(1)),dot(m1.getRow(0), m2.getCol(2)), dot(m1.getRow(0), m2.getCol(3))),
        ray::vec4(dot(m1.getRow(1), m2.getCol(0)), dot(m1.getRow(1), m2.getCol(1)),dot(m1.getRow(1), m2.getCol(2)),dot(m1.getRow(1), m2.getCol(3))),
        ray::vec4(dot(m1.getRow(2), m2.getCol(0)), dot(m1.getRow(2), m2.getCol(1)),dot(m1.getRow(2), m2.getCol(2)),dot(m1.getRow(1), m2.getCol(3))),
        ray::vec4(dot(m1.getRow(3), m2.getCol(0)), dot(m1.getRow(3), m2.getCol(1)),dot(m1.getRow(3), m2.getCol(2)),dot(m1.getRow(3), m2.getCol(3)))};
    }

    __device__ __host__
    inline ray::vec4 dot(const mat4x4& m, const ray::vec4& v) {
        return { dot(m.getRow(0), v), dot(m.getRow(1), v),dot(m.getRow(2), v), dot(m.getRow(3), v)};
    }

    __device__ __host__
    inline mat4x4 translate(const ray::vec3& value) {
        return {
            ray::vec4(1.f,0.f,0.f,value.x),
            ray::vec4(0.f,1.f,0.f,value.y),
            ray::vec4(0.f,0.f,1.f,value.z),
            ray::vec4(0.f,0.f,0.f,1.f)
        };
    }

    __device__ __host__
    inline mat4x4 rotate(const ray::vec3& value) {
        const float ca = cosf(value.x); const float sa = sinf(value.x);
        const float cb = cosf(value.y); const float sb = sinf(value.y);
        const float cg = cosf(value.z); const float sg = sinf(value.z);

        return {
            ray::vec4(ca * cb, (ca * sb * sg) - (sa * cg),(ca * sb * sg) + (sa * sg), 0.f),
            ray::vec4(sa * cb, (sa * sb * sg) + (ca * cg),(sa * sb * sg) - (ca * sg), 0.f),
            ray::vec4(-sb, cb*sg, cb * cg, 0.f),
            ray::vec4(0.f,0.f,0.f,1.f)
        };
    }

}
#endif //RAYMARCHINGCUDA_RAY_MATHS_CUH