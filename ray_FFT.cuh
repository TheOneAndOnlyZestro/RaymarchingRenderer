#ifndef RAYMARCHINGCUDA_RAY_FFT_H
#define RAYMARCHINGCUDA_RAY_FFT_H

#include <vector>
#define PI acos(-1.0)

#ifndef EPSILON
#define EPSILON 1.e-6
#endif
#include <cmath>

namespace ray {
    struct complex {
        float a,b;
        __host__ __device__
        complex(const int n, const int m, const int N) {
            assert(N != 0);
            if (n == 0 || m == 0) {
                a =1;
                b =0;
                return;
            }
            const float theta = static_cast<float>(2 * PI * n * m) / static_cast<float>(N);
            a = ( fabsf(cosf(theta)) < EPSILON) ? 0 : cosf(theta);
            b = ( fabsf(sinf(theta)) < EPSILON) ? 0 : sinf(theta);
        }

        __host__ __device__
        complex():a(0.f), b(0.f){}
        __host__ __device__
        complex(const float _a, const float _b): a(_a), b(_b){}

        __host__ __device__
        complex(const float _a): a(_a), b(0.f) {}
    };

    __host__ __device__
    inline complex operator*(const complex& c1, const complex& c2) {
        return complex(c1.a*c2.a - c1.b * c2.b, c1.a*c2.b + c2.a*c1.b);
    }

    __host__ __device__
    inline complex operator+(const complex& c1, const complex& c2) {
        return complex(c1.a + c2.a, c1.b + c2.b);
    }
    __host__ __device__
    inline complex operator-(const complex& c1, const complex& c2) {
        return complex(c1.a - c2.a, c1.b - c2.b);
    }

    __host__ __device__
    inline complex operator*(const complex& c, const float scalar) {
        return complex(c.a*scalar, c.b*scalar);
    }

    __host__ __device__
    inline complex operator/(const complex& c, const float scalar) {
        return complex(c.a/scalar, c.b/scalar);
    }

    __host__ __device__
    inline complex conjugate(const complex& c) {
        return complex(c.a, -c.b);
    }
    __host__ __device__
    inline complex operator~(const complex& c) {
        return conjugate(c);
    }

    __host__ __device__
    inline float length(const complex& c) {
        return sqrtf((c * ~c).a);
    }

    __host__ __device__
    inline float angle(const complex& c) {
        return atan2f(c.b, c.a);
    }

    __host__ __device__
    inline complex operator/(const complex& c1, const complex& c2) {
        return (c1 * ~c2) / ((c2 * ~c2).a);
    }
}

//FFT implementation
namespace ray {

    inline std::vector<std::vector<complex>> seperateEvenAndOdd(const std::vector<complex>& f) {
        std::vector<std::vector<complex>> result(2);
        result[0] = std::vector<complex>(f.size()/2);//even
        result[1] = std::vector<complex>(f.size()/2);//odd
        for (unsigned int i =0; i < f.size()/2; i++) {
            (result[0])[i] = f[2*i];
            (result[1])[i] = f[(2*i)+1];
        }
        return result;
    }

    inline std::vector<complex> FFTI(const std::vector<complex>& f, const bool inverse = false) {
        //Result vector
        std::vector<complex> fourier(f.size());

        //Anchor of recursion
        if (f.size() == 1){return f;}

        //Divide the even and odd indices
        std::vector<std::vector<complex>> split = seperateEvenAndOdd(f);

        //Do recursive step
        std::vector<complex> even = FFTI(split[0], inverse);
        std::vector<complex> odd = FFTI(split[1], inverse);

        //Assuming that indeed we have the even and odd, we can now calculate current f
        for (int i =0; i < f.size()/2; i++) {
            complex w = inverse ? complex(1,i,f.size()) : complex(1,-i,f.size());
            fourier[i] = even[i] + (w * odd[i]) ;
            fourier[i + f.size()/2] = even[i] - (w * odd[i]);
        }

        return fourier;
    }
    inline std::vector<complex> FFT(const std::vector<complex>& f, const bool inverse = false) {
        std::vector<complex> c = FFTI(f, inverse);
        if (inverse) {
            for (unsigned int i =0; i < f.size(); i++) {
                c[i] = c[i] / c.size();
            }
        }
        return c;
    }
}
#endif //RAYMARCHINGCUDA_RAY_FFT_H