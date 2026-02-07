#pragma once
#include <iostream>
#include <vector>
#include "vec4.h"

struct alignas(16) MatData {
    union {
        alignas(16) float m[4][4];
        alignas(16) float a[16];
    };

    void zero() {
        __m128 z = _mm_setzero_ps();
        _mm_store_ps(&a[0], z);
        _mm_store_ps(&a[4], z);
        _mm_store_ps(&a[8], z);
        _mm_store_ps(&a[12], z);
    }

    void identity() {
        _mm_store_ps(&a[0], _mm_set_ps(0, 0, 0, 1));
        _mm_store_ps(&a[4], _mm_set_ps(0, 0, 1, 0));
        _mm_store_ps(&a[8], _mm_set_ps(0, 1, 0, 0));
        _mm_store_ps(&a[12], _mm_set_ps(1, 0, 0, 0));
    }
};

struct Matrix_Mul_Scalar {
    static inline void mul(MatData& out, const MatData& A, const MatData& B) {
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                out.a[r * 4 + c] =
                    A.a[r * 4 + 0] * B.a[0 * 4 + c] +
                    A.a[r * 4 + 1] * B.a[1 * 4 + c] +
                    A.a[r * 4 + 2] * B.a[2 * 4 + c] +
                    A.a[r * 4 + 3] * B.a[3 * 4 + c];
            }
        }
    }

    static inline void mul(vec4& out, const MatData& A, const vec4& v) {
        out[0] = A.a[0] * v[0] + A.a[1] * v[1] + A.a[2] * v[2] + A.a[3] * v[3];
        out[1] = A.a[4] * v[0] + A.a[5] * v[1] + A.a[6] * v[2] + A.a[7] * v[3];
        out[2] = A.a[8] * v[0] + A.a[9] * v[1] + A.a[10] * v[2] + A.a[11] * v[3];
        out[3] = A.a[12] * v[0] + A.a[13] * v[1] + A.a[14] * v[2] + A.a[15] * v[3];
    }
};



struct Matrix_Mul_SIMD {
    static inline void mul(MatData& out, const MatData& A, const MatData& B) {
        __m128 b0 = _mm_load_ps(&B.a[0]);
        __m128 b1 = _mm_load_ps(&B.a[4]);
        __m128 b2 = _mm_load_ps(&B.a[8]);
        __m128 b3 = _mm_load_ps(&B.a[12]);

        for (int r = 0; r < 4; ++r) {
            __m128 res = _mm_mul_ps(_mm_set1_ps(A.a[r * 4 + 0]), b0);
            res = _mm_fmadd_ps(_mm_set1_ps(A.a[r * 4 + 1]), b1, res);
            res = _mm_fmadd_ps(_mm_set1_ps(A.a[r * 4 + 2]), b2, res);
            res = _mm_fmadd_ps(_mm_set1_ps(A.a[r * 4 + 3]), b3, res);
            _mm_store_ps(&out.a[r * 4], res);
        }
    }

    static inline void mul(vec4& out, const MatData& A, const vec4& v) {
        __m128 vec = _mm_load_ps(&v[0]); // [v0 v1 v2 v3]
        for (int row = 0; row < 4; ++row) {
            __m128 matRow = _mm_load_ps(&A.a[row * 4]); // matrix row
            __m128 mul = _mm_mul_ps(matRow, vec); // element-wise multiply
            // horizontal add: (((x+y)+z)+w)
            __m128 sum = _mm_hadd_ps(mul, mul);
            sum = _mm_hadd_ps(sum, sum);
            out[row] = _mm_cvtss_f32(sum);
        }
    }
};

struct Matrix_Mul_SIMD_DP {
    // SSE matrix multiply: out = A * B
    static inline void mul(MatData& out, const MatData& A, const MatData& B) {
        __m128 b0 = _mm_load_ps(&B.a[0]);
        __m128 b1 = _mm_load_ps(&B.a[4]);
        __m128 b2 = _mm_load_ps(&B.a[8]);
        __m128 b3 = _mm_load_ps(&B.a[12]);

        for (int r = 0; r < 4; ++r) {
            __m128 a0 = _mm_set1_ps(A.a[r * 4 + 0]);
            __m128 a1 = _mm_set1_ps(A.a[r * 4 + 1]);
            __m128 a2 = _mm_set1_ps(A.a[r * 4 + 2]);
            __m128 a3 = _mm_set1_ps(A.a[r * 4 + 3]);

            __m128 res0 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(a0, b0),
                _mm_mul_ps(a1, b1)),
                _mm_add_ps(_mm_mul_ps(a2, b2),
                    _mm_mul_ps(a3, b3)));

            _mm_store_ps(&out.a[r * 4], res0);
        }
    }

    // SSE matrix * vector using _mm_dp_ps
    static inline void mul(vec4& out, const MatData& A, const vec4& v) {
        __m128 vec = _mm_load_ps(&v[0]); // [v0 v1 v2 v3]

        for (int row = 0; row < 4; ++row) {
            __m128 matRow = _mm_load_ps(&A.a[row * 4]); // load row

            // SSE4.1 dot product: dp of 4 elements -> result in lowest float
            __m128 dp = _mm_dp_ps(matRow, vec, 0xF1); // mask 0xF1: multiply all, store in lowest
            out[row] = _mm_cvtss_f32(dp);            // extract lowest float
        }
    }
};

struct Matrix_Mul_SIMD_DP2 {
    // SSE matrix multiply: out = A * B
    static inline void mul(MatData& out, const MatData& A, const MatData& B) {
        __m128 rowVec1 = _mm_loadu_ps(&A.a[0]);
        __m128 rowVec2 = _mm_loadu_ps(&A.a[4]);
        __m128 rowVec3 = _mm_loadu_ps(&A.a[8]);
        __m128 rowVec4 = _mm_loadu_ps(&A.a[12]);


        for (int col = 0; col < 4; ++col) {

            __m128 colVec = _mm_set_ps(B.a[3 * 4 + col], B.a[2 * 4 + col], B.a[1 * 4 + col], B.a[0 * 4 + col]);

            out.a[0 * 4 + col] = _mm_cvtss_f32(_mm_dp_ps(rowVec1, colVec, 0xF1));
            out.a[1 * 4 + col] = _mm_cvtss_f32(_mm_dp_ps(rowVec2, colVec, 0xF1));
            out.a[2 * 4 + col] = _mm_cvtss_f32(_mm_dp_ps(rowVec3, colVec, 0xF1));
            out.a[3 * 4 + col] = _mm_cvtss_f32(_mm_dp_ps(rowVec4, colVec, 0xF1));
            //a[row * 4 + 0] * mx.a[0 * 4 + col] +a[row * 4 + 1] * mx.a[1 * 4 + col] + a[row * 4 + 2] * mx.a[2 * 4 + col] +a[row * 4 + 3] * mx.a[3 * 4 + col];
        }
    }

    // SSE matrix * vector using _mm_dp_ps
    static inline void mul(vec4& out, const MatData& A, const vec4& v) {
        __m128 value = _mm_loadu_ps(v.v);
        __m128 row0 = _mm_loadu_ps(&A.a[0]);
        __m128 row1 = _mm_loadu_ps(&A.a[4]);
        __m128 row2 = _mm_loadu_ps(&A.a[8]);
        __m128 row3 = _mm_loadu_ps(&A.a[12]);
        out.v[0] = _mm_cvtss_f32(_mm_dp_ps(row0, value, 0xF1));//dot product 
        out.v[1] = _mm_cvtss_f32(_mm_dp_ps(row1, value, 0xF1));
        out.v[2] = _mm_cvtss_f32(_mm_dp_ps(row2, value, 0xF1));
        out.v[3] = _mm_cvtss_f32(_mm_dp_ps(row3, value, 0xF1));
    }
};


struct Matrix_Mul_AVX2 {
    static inline void mul(MatData& out, const MatData& A, const MatData& B) {
        __m256 b0 = _mm256_loadu_ps(&B.a[0]);
        __m256 b1 = _mm256_loadu_ps(&B.a[8]);

        for (int r = 0; r < 4; ++r) {
            // Broadcast A row elements
            __m256 a0 = _mm256_set1_ps(A.a[r * 4 + 0]);
            __m256 a1 = _mm256_set1_ps(A.a[r * 4 + 1]);
            __m256 a2 = _mm256_set1_ps(A.a[r * 4 + 2]);
            __m256 a3 = _mm256_set1_ps(A.a[r * 4 + 3]);

            // Multiply & add
            __m256 res = _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(a0, b0), _mm256_mul_ps(a1, b0)),
                _mm256_add_ps(_mm256_mul_ps(a2, b1), _mm256_mul_ps(a3, b1))
            );

            // Horizontal sum for each 128-bit lane
            __m128 lo = _mm256_castps256_ps128(res);
            __m128 hi = _mm256_extractf128_ps(res, 1);
            __m128 sum = _mm_add_ps(lo, hi);

            _mm_store_ps(&out.a[r * 4], sum);
        }
    }

    static inline void mul(vec4& out, const MatData& A, const vec4& v) {
        __m128 vec = _mm_loadu_ps(&v[0]); // [v0 v1 v2 v3]
        for (int row = 0; row < 4; ++row) {
            __m128 mrow = _mm_loadu_ps(&A.a[row * 4]);
            __m128 mul = _mm_mul_ps(mrow, vec);
            __m128 sum = _mm_hadd_ps(mul, mul);
            sum = _mm_hadd_ps(sum, sum);
            out[row] = _mm_cvtss_f32(sum);
        }
    }
};

struct Matrix_Trans_Intial {
    static inline void makePerspective(MatData& out, float fov, float aspect, float n, float f) {
        out.zero();
        float tanHalfFov = std::tan(fov / 2.0f);
        out.a[0] = 1.0f / (aspect * tanHalfFov);
        out.a[5] = 1.0f / tanHalfFov;
        out.a[10] = -f / (f - n);
        out.a[11] = -(f * n) / (f - n);
        out.a[14] = -1.0f;
    }

    static inline void makeTranslation(MatData& out, float tx, float ty, float tz) {
        out.identity();
        out.a[3] = tx;
        out.a[7] = ty;
        out.a[11] = tz;
    }

    static inline void makeRotateZ(MatData& out, float aRad) {
        out.identity();
        out.a[0] = std::cos(aRad);
        out.a[1] = -std::sin(aRad);
        out.a[4] = std::sin(aRad);
        out.a[5] = std::cos(aRad);
    }

    static inline void makeRotateX(MatData& out, float aRad) {
        out.identity();
        out.a[5] = std::cos(aRad);
        out.a[6] = -std::sin(aRad);
        out.a[9] = std::sin(aRad);
        out.a[10] = std::cos(aRad);
    }

     static inline void makeRotateY(MatData& out, float aRad) {
         out.identity();
         out.a[0] = std::cos(aRad);
         out.a[2] = std::sin(aRad);
         out.a[8] = -std::sin(aRad);
         out.a[10] = std::cos(aRad);
    }

    static inline void makeScale(MatData& out, float s) {
        s = std::max(s, 0.01f); // Ensure scaling factor is not too small
        out.identity();
        out.a[0] = s;
        out.a[5] = s;
        out.a[10] = s;
    }

    static inline void makeIdentity(MatData& out) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                out.m[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
};

struct Matrix_Trans_Manually {
    static inline void makePerspective(MatData& out, float fov, float aspect, float n, float f) {
        const float tanHalf = std::tan(fov * 0.5f);
        const float invTan = 1.0f / tanHalf;
        const float invFN = 1.0f / (f - n);

        out.a[0] = invTan / aspect;
        out.a[1] = 0.f;
        out.a[2] = 0.f;
        out.a[3] = 0.f;

        out.a[4] = 0.f;
        out.a[5] = invTan;
        out.a[6] = 0.f;
        out.a[7] = 0.f;

        out.a[8] = 0.f;
        out.a[9] = 0.f;
        out.a[10] = -f * invFN;
        out.a[11] = -(f * n) * invFN;

        out.a[12] = 0.f;
        out.a[13] = 0.f;
        out.a[14] = -1.f;
        out.a[15] = 0.f;
    }


    static inline void makeTranslation(MatData& out, float tx, float ty, float tz) {
        out.a[0] = 1.f;
        out.a[1] = 0.f;
        out.a[2] = 0.f;
        out.a[3] = tx;

        out.a[4] = 0.f;
        out.a[5] = 1.f;
        out.a[6] = 0.f; 
        out.a[7] = ty;
        
        out.a[8] = 0.f; 
        out.a[9] = 0.f; 
        out.a[10] = 1.f; 
        out.a[11] = tz;
        
        out.a[12] = 0.f;
        out.a[13] = 0.f; 
        out.a[14] = 0.f; 
        out.a[15] = 1.f;
    }

    static inline void makeRotateZ(MatData& out, float aRad)
    {
        float c = std::cos(aRad);
        float s = std::sin(aRad);

        out.a[0] = c;   out.a[1] = -s;   out.a[2] = 0.f; out.a[3] = 0.f;
        out.a[4] = s;   out.a[5] = c;   out.a[6] = 0.f; out.a[7] = 0.f;
        out.a[8] = 0.f;  out.a[9] = 0.f;  out.a[10] = 1.f; out.a[11] = 0.f;
        out.a[12] = 0.f;  out.a[13] = 0.f;  out.a[14] = 0.f; out.a[15] = 1.f;
    }


    static inline void makeRotateX(MatData& out, float aRad)
    {
        float c = std::cos(aRad);
        float s = std::sin(aRad);

        out.a[0] = 1.f; out.a[1] = 0.f; out.a[2] = 0.f; out.a[3] = 0.f;
        out.a[4] = 0.f; out.a[5] = c;  out.a[6] = -s;  out.a[7] = 0.f;
        out.a[8] = 0.f; out.a[9] = s;  out.a[10] = c;  out.a[11] = 0.f;
        out.a[12] = 0.f; out.a[13] = 0.f; out.a[14] = 0.f; out.a[15] = 1.f;
    }


    static inline void makeRotateY(MatData& out, float aRad)
    {
        float c = std::cos(aRad);
        float s = std::sin(aRad);

        out.a[0] = c;  out.a[1] = 0.f; out.a[2] = s;  out.a[3] = 0.f;
        out.a[4] = 0.f; out.a[5] = 1.f; out.a[6] = 0.f; out.a[7] = 0.f;
        out.a[8] = -s;  out.a[9] = 0.f; out.a[10] = c;  out.a[11] = 0.f;
        out.a[12] = 0.f; out.a[13] = 0.f; out.a[14] = 0.f; out.a[15] = 1.f;
    }


    static inline void makeScale(MatData& out, float s)
    {
        s = std::max(s, 0.01f);

        out.a[0] = s;   out.a[1] = 0.f; out.a[2] = 0.f; out.a[3] = 0.f;
        out.a[4] = 0.f; out.a[5] = s;   out.a[6] = 0.f; out.a[7] = 0.f;
        out.a[8] = 0.f; out.a[9] = 0.f; out.a[10] = s;   out.a[11] = 0.f;
        out.a[12] = 0.f; out.a[13] = 0.f; out.a[14] = 0.f; out.a[15] = 1.f;
    }


    static inline void makeIdentity(MatData& out)
    {
        out.a[0] = 1.f; out.a[1] = 0.f; out.a[2] = 0.f; out.a[3] = 0.f;
        out.a[4] = 0.f; out.a[5] = 1.f; out.a[6] = 0.f; out.a[7] = 0.f;
        out.a[8] = 0.f; out.a[9] = 0.f; out.a[10] = 1.f; out.a[11] = 0.f;
        out.a[12] = 0.f; out.a[13] = 0.f; out.a[14] = 0.f; out.a[15] = 1.f;
    }

    static inline void makeRotateXYZ(MatData& out,
        float x, float y, float z)
    {
        float cx = std::cos(x), sx = std::sin(x);
        float cy = std::cos(y), sy = std::sin(y);
        float cz = std::cos(z), sz = std::sin(z);

        out.a[0] = cy * cz;
        out.a[1] = -cy * sz;
        out.a[2] = sy;
        out.a[3] = 0.f;

        out.a[4] = sx * sy * cz + cx * sz;
        out.a[5] = -sx * sy * sz + cx * cz;
        out.a[6] = -sx * cy;
        out.a[7] = 0.f;

        out.a[8] = -cx * sy * cz + sx * sz;
        out.a[9] = cx * sy * sz + sx * cz;
        out.a[10] = cx * cy;
        out.a[11] = 0.f;

        out.a[12] = 0.f;
        out.a[13] = 0.f;
        out.a[14] = 0.f;
        out.a[15] = 1.f;
    }

};
/*
struct Matrix_Trans_SIMD {
    static inline void makePerspective(MatData& out, float fov, float aspect, float n, float f) {
        const float tanHalf = std::tan(fov * 0.5f);
        const float invTan = 1.0f / tanHalf;
        const float invFN = 1.0f / (f - n);

        _mm_store_ps(&out.a[0], _mm_set_ps(0.f, 0.f, 0.f, invTan / aspect));

        _mm_store_ps(&out.a[4], _mm_set_ps(0.f, 0.f, invTan, 0.f));

        _mm_store_ps(&out.a[8], _mm_set_ps(-(f * n) * invFN, -f * invFN, 0.f, 0.f));

        _mm_store_ps(&out.a[12], _mm_set_ps(0.f, -1.f, 0.f, 0.f));
    }



    static inline void makeTranslation(MatData& out, float tx, float ty, float tz) {
        _mm_store_ps(&out.a[0], _mm_set_ps(tx, 0.f, 0.f, 1.f));
        _mm_store_ps(&out.a[4], _mm_set_ps(ty, 0.f, 1.f, 0.f));
        _mm_store_ps(&out.a[8], _mm_set_ps(tz, 1.f, 0.f, 0.f));
        _mm_store_ps(&out.a[12], _mm_set_ps(1.f, 0.f, 0.f, 0.f));
    }

    static inline void makeRotateZ(MatData& out, float aRad) {
        float c = std::cos(aRad);
        float s = std::sin(aRad);

        _mm_store_ps(&out.a[0], _mm_set_ps(0.f, 0.f, -s, c));
        _mm_store_ps(&out.a[4], _mm_set_ps(0.f, 0.f, c, s));
        _mm_store_ps(&out.a[8], _mm_set_ps(0.f, 1.f, 0.f, 0.f));
        _mm_store_ps(&out.a[12], _mm_set_ps(1.f, 0.f, 0.f, 0.f));

    }

    static inline void makeRotateX(MatData& out, float aRad) {
        float c = std::cos(aRad);
        float s = std::sin(aRad);

        _mm_store_ps(&out.a[0], _mm_set_ps(0.f, 0.f, 0.f, 1.f));
        _mm_store_ps(&out.a[4], _mm_set_ps(0.f, -s, c, 0.f));
        _mm_store_ps(&out.a[8], _mm_set_ps(0.f, c, s, 0.f));
        _mm_store_ps(&out.a[12], _mm_set_ps(1.f, 0.f, 0.f, 0.f));
    }

    static inline void makeRotateY(MatData& out, float aRad) {
        float c = std::cos(aRad);
        float s = std::sin(aRad);

        _mm_store_ps(&out.a[0], _mm_set_ps(0.f, s, 0.f, c));
        _mm_store_ps(&out.a[4], _mm_set_ps(0.f, 0.f, 1.f, 0.f));
        _mm_store_ps(&out.a[8], _mm_set_ps(0.f, c, 0.f, -s));
        _mm_store_ps(&out.a[12], _mm_set_ps(1.f, 0.f, 0.f, 0.f));
    }

    static inline void makeScale(MatData& out, float s) {
        s = std::max(s, 0.01f);

        _mm_store_ps(&out.a[0], _mm_set_ps(0.f, 0.f, 0.f, s));
        _mm_store_ps(&out.a[4], _mm_set_ps(0.f, 0.f, s, 0.f));
        _mm_store_ps(&out.a[8], _mm_set_ps(0.f, s, 0.f, 0.f));
        _mm_store_ps(&out.a[12], _mm_set_ps(1.f, 0.f, 0.f, 0.f));
    }

    static inline void makeIdentity(MatData& out) {
        _mm_store_ps(&out.a[0], _mm_set_ps(0.f, 0.f, 0.f, 1.f));
        _mm_store_ps(&out.a[4], _mm_set_ps(0.f, 0.f, 1.f, 0.f));
        _mm_store_ps(&out.a[8], _mm_set_ps(0.f, 1.f, 0.f, 0.f));
        _mm_store_ps(&out.a[12], _mm_set_ps(1.f, 0.f, 0.f, 0.f));
    }

    static inline void makeRotateXYZ(MatData& out,
        float x, float y, float z)
    {
        float cx = std::cos(x), sx = std::sin(x);
        float cy = std::cos(y), sy = std::sin(y);
        float cz = std::cos(z), sz = std::sin(z);

        _mm_store_ps(&out.a[0],
            _mm_set_ps(0.f,
                sy,
                -cy * sz,
                cy * cz));

        _mm_store_ps(&out.a[4],
            _mm_set_ps(0.f,
                -sx * cy,
                -sx * sy * sz + cx * cz,
                sx * sy * cz + cx * sz));

        _mm_store_ps(&out.a[8],
            _mm_set_ps(0.f,
                cx * cy,
                cx * sy * sz + sx * cz,
                -cx * sy * cz + sx * sz));

        _mm_store_ps(&out.a[12],
            _mm_set_ps(1.f, 0.f, 0.f, 0.f));
    }

};
*/
struct Matrix_Trans_SIMD {

    static inline void makePerspective(MatData& out,
        float fov, float aspect,
        float n, float f)
    {
        const float tanHalf = std::tan(fov * 0.5f);
        const float invTan = 1.0f / tanHalf;
        const float invFN = 1.0f / (f - n);

        _mm_store_ps(&out.a[0],
            _mm_setr_ps(invTan / aspect, 0.f, 0.f, 0.f));

        _mm_store_ps(&out.a[4],
            _mm_setr_ps(0.f, invTan, 0.f, 0.f));

        _mm_store_ps(&out.a[8],
            _mm_setr_ps(0.f, 0.f, -f * invFN, -(f * n) * invFN));

        _mm_store_ps(&out.a[12],
            _mm_setr_ps(0.f, 0.f, -1.f, 0.f));
    }

    static inline void makeTranslation(MatData& out,
        float tx, float ty, float tz)
    {
        _mm_store_ps(&out.a[0],
            _mm_setr_ps(1.f, 0.f, 0.f, tx));

        _mm_store_ps(&out.a[4],
            _mm_setr_ps(0.f, 1.f, 0.f, ty));

        _mm_store_ps(&out.a[8],
            _mm_setr_ps(0.f, 0.f, 1.f, tz));

        _mm_store_ps(&out.a[12],
            _mm_setr_ps(0.f, 0.f, 0.f, 1.f));
    }

    static inline void makeRotateZ(MatData& out, float aRad)
    {
        float c = std::cos(aRad);
        float s = std::sin(aRad);

        _mm_store_ps(&out.a[0],
            _mm_setr_ps(c, -s, 0.f, 0.f));

        _mm_store_ps(&out.a[4],
            _mm_setr_ps(s, c, 0.f, 0.f));

        _mm_store_ps(&out.a[8],
            _mm_setr_ps(0.f, 0.f, 1.f, 0.f));

        _mm_store_ps(&out.a[12],
            _mm_setr_ps(0.f, 0.f, 0.f, 1.f));
    }

    static inline void makeRotateX(MatData& out, float aRad)
    {
        float c = std::cos(aRad);
        float s = std::sin(aRad);

        _mm_store_ps(&out.a[0],
            _mm_setr_ps(1.f, 0.f, 0.f, 0.f));

        _mm_store_ps(&out.a[4],
            _mm_setr_ps(0.f, c, -s, 0.f));

        _mm_store_ps(&out.a[8],
            _mm_setr_ps(0.f, s, c, 0.f));

        _mm_store_ps(&out.a[12],
            _mm_setr_ps(0.f, 0.f, 0.f, 1.f));
    }

    static inline void makeRotateY(MatData& out, float aRad)
    {
        float c = std::cos(aRad);
        float s = std::sin(aRad);

        _mm_store_ps(&out.a[0],
            _mm_setr_ps(c, 0.f, s, 0.f));

        _mm_store_ps(&out.a[4],
            _mm_setr_ps(0.f, 1.f, 0.f, 0.f));

        _mm_store_ps(&out.a[8],
            _mm_setr_ps(-s, 0.f, c, 0.f));

        _mm_store_ps(&out.a[12],
            _mm_setr_ps(0.f, 0.f, 0.f, 1.f));
    }

    static inline void makeScale(MatData& out, float s)
    {
        s = std::max(s, 0.01f);

        _mm_store_ps(&out.a[0],
            _mm_setr_ps(s, 0.f, 0.f, 0.f));

        _mm_store_ps(&out.a[4],
            _mm_setr_ps(0.f, s, 0.f, 0.f));

        _mm_store_ps(&out.a[8],
            _mm_setr_ps(0.f, 0.f, s, 0.f));

        _mm_store_ps(&out.a[12],
            _mm_setr_ps(0.f, 0.f, 0.f, 1.f));
    }

    static inline void makeIdentity(MatData& out)
    {
        _mm_store_ps(&out.a[0],
            _mm_setr_ps(1.f, 0.f, 0.f, 0.f));

        _mm_store_ps(&out.a[4],
            _mm_setr_ps(0.f, 1.f, 0.f, 0.f));

        _mm_store_ps(&out.a[8],
            _mm_setr_ps(0.f, 0.f, 1.f, 0.f));

        _mm_store_ps(&out.a[12],
            _mm_setr_ps(0.f, 0.f, 0.f, 1.f));
    }

    static inline void makeRotateXYZ(MatData& out,
        float x, float y, float z)
    {
        float cx = std::cos(x), sx = std::sin(x);
        float cy = std::cos(y), sy = std::sin(y);
        float cz = std::cos(z), sz = std::sin(z);

        _mm_store_ps(&out.a[0],
            _mm_setr_ps(
                cy * cz,
                -cy * sz,
                sy,
                0.f));

        _mm_store_ps(&out.a[4],
            _mm_setr_ps(
                sx * sy * cz + cx * sz,
                -sx * sy * sz + cx * cz,
                -sx * cy,
                0.f));

        _mm_store_ps(&out.a[8],
            _mm_setr_ps(
                -cx * sy * cz + sx * sz,
                cx * sy * sz + sx * cz,
                cx * cy,
                0.f));

        _mm_store_ps(&out.a[12],
            _mm_setr_ps(0.f, 0.f, 0.f, 1.f));
    }
};

template<typename MulPolicy = Matrix_Mul_Scalar, typename TransPolicy = Matrix_Trans_Intial>
class alignas(16) matrixT {
public:
    MatData data;

    matrixT() { data.identity(); }

    float& operator()(int r, int c) { return data.m[r][c]; }
    const float& operator()(int r, int c) const { return data.m[r][c]; }

    void display() const {
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c)
                std::cout << data.m[r][c] << '\t';
            std::cout << '\n';
        }
    }

    // matrix * matrix
    matrixT operator*(const matrixT& rhs) const {
        matrixT r;
        MulPolicy::mul(r.data, data, rhs.data);
        return r;
    }

    vec4 operator*(const vec4& v) const {
        vec4 r;
        MulPolicy::mul(r, data, v);
        return r;
    }

    static matrixT makePerspective(float fov, float aspect, float n, float f) {
        matrixT m;
        TransPolicy::makePerspective(m.data, fov, aspect, n, f);
        return m;
    }

    static matrixT makeTranslation(float tx, float ty, float tz) {
        matrixT m;
        TransPolicy::makeTranslation(m.data, tx, ty, tz);
        return m;
    }

    static matrixT makeRotateZ(float aRad) {
        matrixT m;
        TransPolicy::makeRotateZ(m.data, aRad);
        return m;
    }

    static matrixT makeRotateX(float aRad) {
        matrixT m;
        TransPolicy::makeRotateX(m.data, aRad);
        return m;
    }

    static matrixT makeRotateY(float aRad) {
        matrixT m;
        TransPolicy::makeRotateY(m.data, aRad);
        return m;
    }

    //static matrixT makeRotateXYZ(float x, float y, float z) {
    //    matrixT m;
    //    TransPolicy::makeRotateXYZ(m.data, x, y, z);
    //    return m;
    //}

    static matrixT makeRotateXYZ(float x, float y, float z)
    {
        matrixT m;
        float cx = std::cos(x), sx = std::sin(x);
        float cy = std::cos(y), sy = std::sin(y);
        float cz = std::cos(z), sz = std::sin(z);

        m.data.a[0] = cy * cz;
        m.data.a[1] = -cy * sz;
        m.data.a[2] = sy;
        m.data.a[3] = 0.f;

        m.data.a[4] = sx * sy * cz + cx * sz;
        m.data.a[5] = -sx * sy * sz + cx * cz;
        m.data.a[6] = -sx * cy;
        m.data.a[7] = 0.f;

        m.data.a[8] = -cx * sy * cz + sx * sz;
        m.data.a[9] = cx * sy * sz + sx * cz;
        m.data.a[10] = cx * cy;
        m.data.a[11] = 0.f;

        m.data.a[12] = 0.f;
        m.data.a[13] = 0.f;
        m.data.a[14] = 0.f;
        m.data.a[15] = 1.f;

        return m;
    }

    static matrixT makeScale(float s) {
        matrixT m;
        TransPolicy::makeScale(m.data, s);
        return m;
    }

    static matrixT makeIdentity() {
        matrixT m;
        TransPolicy::makeIdentity(m.data);
        return m;
    }

    static inline void batchMul(
        const MatData& A,
        const float* px, const float* py, const float* pz, const float* pw,
        float* outX, float* outY, float* outZ, float* outW)
    {
        // Load matrix rows
        __m256 row0 = _mm256_set_m128(_mm_load_ps(&A.a[12]), _mm_load_ps(&A.a[0]));
        __m256 row1 = _mm256_set_m128(_mm_load_ps(&A.a[13]), _mm_load_ps(&A.a[4]));
        __m256 row2 = _mm256_set_m128(_mm_load_ps(&A.a[14]), _mm_load_ps(&A.a[8]));
        __m256 row3 = _mm256_set_m128(_mm_load_ps(&A.a[15]), _mm_load_ps(&A.a[12])); // w row

        // Load 8 vertices
        __m256 vx = _mm256_loadu_ps(px);
        __m256 vy = _mm256_loadu_ps(py);
        __m256 vz = _mm256_loadu_ps(pz);
        __m256 vw = _mm256_loadu_ps(pw);

        // Compute outX = row0 * vec
        __m256 outx = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&A.a[0]), vx),
                _mm256_mul_ps(_mm256_broadcast_ss(&A.a[1]), vy)),
            _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&A.a[2]), vz),
                _mm256_mul_ps(_mm256_broadcast_ss(&A.a[3]), vw))
        );

        __m256 outy = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&A.a[4]), vx),
                _mm256_mul_ps(_mm256_broadcast_ss(&A.a[5]), vy)),
            _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&A.a[6]), vz),
                _mm256_mul_ps(_mm256_broadcast_ss(&A.a[7]), vw))
        );

        __m256 outz = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&A.a[8]), vx),
                _mm256_mul_ps(_mm256_broadcast_ss(&A.a[9]), vy)),
            _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&A.a[10]), vz),
                _mm256_mul_ps(_mm256_broadcast_ss(&A.a[11]), vw))
        );

        __m256 outw = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&A.a[12]), vx),
                _mm256_mul_ps(_mm256_broadcast_ss(&A.a[13]), vy)),
            _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&A.a[14]), vz),
                _mm256_mul_ps(_mm256_broadcast_ss(&A.a[15]), vw))
        );

        // Store results
        _mm256_storeu_ps(outX, outx);
        _mm256_storeu_ps(outY, outy);
        _mm256_storeu_ps(outZ, outz);
        _mm256_storeu_ps(outW, outw);
    }
};
