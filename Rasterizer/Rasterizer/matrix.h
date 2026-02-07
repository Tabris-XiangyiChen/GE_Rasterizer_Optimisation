#pragma once

#include <iostream>
#include <vector>
#include "vec4.h"

// Matrix class for 4x4 transformation matrices
class alignas(16) matrix {
    union {
        float m[4][4]; // 2D array representation of the matrix
        float a[16];   // 1D array representation of the matrix for linear access
    };

public:
    // Default constructor initializes the matrix as an identity matrix
    matrix() {
        identity();
    }

    // Access matrix elements by row and column
    float& operator()(unsigned int row, unsigned int col) { return m[row][col]; }

    // Display the matrix elements in a readable format
    void display() {
        for (unsigned int i = 0; i < 4; i++) {
            for (unsigned int j = 0; j < 4; j++)
                std::cout << m[i][j] << '\t';
            std::cout << std::endl;
        }
    }

    // Multiply the matrix by a 4D vector
    // Input Variables:
    // - v: vec4 object to multiply with the matrix
    // Returns the resulting transformed vec4
    vec4 operator * (const vec4& v) const {
        vec4 result;
        result[0] = a[0] * v[0] + a[1] * v[1] + a[2] * v[2] + a[3] * v[3];
        result[1] = a[4] * v[0] + a[5] * v[1] + a[6] * v[2] + a[7] * v[3];
        result[2] = a[8] * v[0] + a[9] * v[1] + a[10] * v[2] + a[11] * v[3];
        result[3] = a[12] * v[0] + a[13] * v[1] + a[14] * v[2] + a[15] * v[3];
        return result;
    }
    // Opt: SIMD SSE Best
    //vec4 operator*(const vec4& v) const {
    //    vec4 result;
    //    __m128 vec = _mm_load_ps(&v[0]); // [v0 v1 v2 v3]
    //    for (int row = 0; row < 4; ++row) {
    //        __m128 matRow = _mm_load_ps(&a[row * 4]); // matrix row
    //        __m128 mul = _mm_mul_ps(matRow, vec); // element-wise multiply
    //        // horizontal add: (((x+y)+z)+w)
    //        __m128 sum = _mm_hadd_ps(mul, mul);
    //        sum = _mm_hadd_ps(sum, sum);
    //        result[row] = _mm_cvtss_f32(sum);
    //    }
    //    return result;
    //}
    
    // Opt: SIMD SSE Unrolling
    //vec4 operator*(const vec4& v) const {
    //    vec4 result;
    //    __m128 vec = _mm_load_ps(&v[0]);
    //    // row 0
    //    {
    //        __m128 r = _mm_load_ps(&a[0]);
    //        __m128 m = _mm_mul_ps(r, vec);
    //        __m128 s = _mm_hadd_ps(m, m);
    //        s = _mm_hadd_ps(s, s);
    //        result[0] = _mm_cvtss_f32(s);
    //    }
    //    // row 1
    //    {
    //        __m128 r = _mm_load_ps(&a[4]);
    //        __m128 m = _mm_mul_ps(r, vec);
    //        __m128 s = _mm_hadd_ps(m, m);
    //        s = _mm_hadd_ps(s, s);
    //        result[1] = _mm_cvtss_f32(s);
    //    }
    //    // row 2
    //    {
    //        __m128 r = _mm_load_ps(&a[8]);
    //        __m128 m = _mm_mul_ps(r, vec);
    //        __m128 s = _mm_hadd_ps(m, m);
    //        s = _mm_hadd_ps(s, s);
    //        result[2] = _mm_cvtss_f32(s);
    //    }
    //    // row 3
    //    {
    //        __m128 r = _mm_load_ps(&a[12]);
    //        __m128 m = _mm_mul_ps(r, vec);
    //        __m128 s = _mm_hadd_ps(m, m);
    //        s = _mm_hadd_ps(s, s);
    //        result[3] = _mm_cvtss_f32(s);
    //    }
    //    return result;
    //}

    // Multiply the matrix by another matrix
    // Input Variables:
    // - mx: Another matrix to multiply with
    // Returns the resulting matrix
    //matrix operator * (const matrix& mx) const {
    //    matrix ret;
    //    for (int row = 0; row < 4; ++row) {
    //        for (int col = 0; col < 4; ++col) {
    //            ret.a[row * 4 + col] =
    //                a[row * 4 + 0] * mx.a[0 * 4 + col] +
    //                a[row * 4 + 1] * mx.a[1 * 4 + col] +
    //                a[row * 4 + 2] * mx.a[2 * 4 + col] +
    //                a[row * 4 + 3] * mx.a[3 * 4 + col];
    //        }
    //    }
    //    return ret;
    //}

    // Opt: one row one time
    //matrix operator*(const matrix& mx) const {
    //    matrix ret;
    //    for (int row = 0; row < 4; ++row) {
    //        const float a0 = a[row * 4 + 0];
    //        const float a1 = a[row * 4 + 1];
    //        const float a2 = a[row * 4 + 2];
    //        const float a3 = a[row * 4 + 3];
    //        for (int col = 0; col < 4; ++col) {
    //            ret.a[row * 4 + col] =
    //                a0 * mx.a[0 * 4 + col] +
    //                a1 * mx.a[1 * 4 + col] +
    //                a2 * mx.a[2 * 4 + col] +
    //                a3 * mx.a[3 * 4 + col];
    //        }
    //    }
    //    return ret;
    //}
    
    // Opt: SIMD SSE
    //matrix operator*(const matrix& mx) const {
    //    matrix ret;
    //    for (int row = 0; row < 4; ++row) {
    //        __m128 r0 = _mm_set1_ps(a[row * 4 + 0]);
    //        __m128 r1 = _mm_set1_ps(a[row * 4 + 1]);
    //        __m128 r2 = _mm_set1_ps(a[row * 4 + 2]);
    //        __m128 r3 = _mm_set1_ps(a[row * 4 + 3]);
    //        __m128 m0 = _mm_load_ps(&mx.a[0]);   // row 0
    //        __m128 m1 = _mm_load_ps(&mx.a[4]);   // row 1
    //        __m128 m2 = _mm_load_ps(&mx.a[8]);   // row 2
    //        __m128 m3 = _mm_load_ps(&mx.a[12]);  // row 3
    //        __m128 res =
    //            _mm_add_ps(
    //                _mm_add_ps(_mm_mul_ps(r0, m0), _mm_mul_ps(r1, m1)),
    //                _mm_add_ps(_mm_mul_ps(r2, m2), _mm_mul_ps(r3, m3))
    //            );
    //        _mm_store_ps(&ret.a[row * 4], res);
    //    }
    //    return ret;
    //}

    // Opt: SIMD SSE Unrolling
    //matrix operator*(const matrix& mx) const {
    //    matrix ret;
    //    //load mx
    //    __m128 m0 = _mm_load_ps(&mx.a[0]);
    //    __m128 m1 = _mm_load_ps(&mx.a[4]);
    //    __m128 m2 = _mm_load_ps(&mx.a[8]);
    //    __m128 m3 = _mm_load_ps(&mx.a[12]);
    //    // row 0
    //    {
    //        __m128 r0 = _mm_set1_ps(a[0]);
    //        __m128 r1 = _mm_set1_ps(a[1]);
    //        __m128 r2 = _mm_set1_ps(a[2]);
    //        __m128 r3 = _mm_set1_ps(a[3]);
    //        __m128 res = _mm_add_ps(
    //            _mm_add_ps(_mm_mul_ps(r0, m0), _mm_mul_ps(r1, m1)),
    //            _mm_add_ps(_mm_mul_ps(r2, m2), _mm_mul_ps(r3, m3))
    //        );
    //        _mm_store_ps(&ret.a[0], res);
    //    }
    //    // row 1
    //    {
    //        __m128 r0 = _mm_set1_ps(a[4]);
    //        __m128 r1 = _mm_set1_ps(a[5]);
    //        __m128 r2 = _mm_set1_ps(a[6]);
    //        __m128 r3 = _mm_set1_ps(a[7]);
    //        __m128 res = _mm_add_ps(
    //            _mm_add_ps(_mm_mul_ps(r0, m0), _mm_mul_ps(r1, m1)),
    //            _mm_add_ps(_mm_mul_ps(r2, m2), _mm_mul_ps(r3, m3))
    //        );
    //        _mm_store_ps(&ret.a[4], res);
    //    }
    //    // row 2
    //    {
    //        __m128 r0 = _mm_set1_ps(a[8]);
    //        __m128 r1 = _mm_set1_ps(a[9]);
    //        __m128 r2 = _mm_set1_ps(a[10]);
    //        __m128 r3 = _mm_set1_ps(a[11]);
    //        __m128 res = _mm_add_ps(
    //            _mm_add_ps(_mm_mul_ps(r0, m0), _mm_mul_ps(r1, m1)),
    //            _mm_add_ps(_mm_mul_ps(r2, m2), _mm_mul_ps(r3, m3))
    //        );
    //        _mm_store_ps(&ret.a[8], res);
    //    }
    //    // row 3
    //    {
    //        __m128 r0 = _mm_set1_ps(a[12]);
    //        __m128 r1 = _mm_set1_ps(a[13]);
    //        __m128 r2 = _mm_set1_ps(a[14]);
    //        __m128 r3 = _mm_set1_ps(a[15]);
    //        __m128 res = _mm_add_ps(
    //            _mm_add_ps(_mm_mul_ps(r0, m0), _mm_mul_ps(r1, m1)),
    //            _mm_add_ps(_mm_mul_ps(r2, m2), _mm_mul_ps(r3, m3))
    //        );
    //        _mm_store_ps(&ret.a[12], res);
    //    }
    //    return ret;
    //}

    // Opt: SIMD FMA SSE Best
    matrix operator*(const matrix& mx) const {
        matrix ret;
        for (int row = 0; row < 4; ++row) {
            __m128 res = _mm_mul_ps(
                _mm_set1_ps(a[row * 4 + 0]),
                _mm_load_ps(&mx.a[0])
            );
            res = _mm_fmadd_ps(
                _mm_set1_ps(a[row * 4 + 1]),
                _mm_load_ps(&mx.a[4]),
                res
            );
            res = _mm_fmadd_ps(
                _mm_set1_ps(a[row * 4 + 2]),
                _mm_load_ps(&mx.a[8]),
                res
            );
            res = _mm_fmadd_ps(
                _mm_set1_ps(a[row * 4 + 3]),
                _mm_load_ps(&mx.a[12]),
                res
            );
            _mm_store_ps(&ret.a[row * 4], res);
        }
        return ret;
    }

    // Opt: SIMD FMA SSE Unrolling
    //matrix operator*(const matrix& mx) const {
    //    matrix ret;
    //    __m128 m0 = _mm_load_ps(&mx.a[0]);
    //    __m128 m1 = _mm_load_ps(&mx.a[4]);
    //    __m128 m2 = _mm_load_ps(&mx.a[8]);
    //    __m128 m3 = _mm_load_ps(&mx.a[12]);
    //    // row 0
    //    __m128 res0 = _mm_mul_ps(_mm_set1_ps(a[0]), m0);
    //    res0 = _mm_fmadd_ps(_mm_set1_ps(a[1]), m1, res0);
    //    res0 = _mm_fmadd_ps(_mm_set1_ps(a[2]), m2, res0);
    //    res0 = _mm_fmadd_ps(_mm_set1_ps(a[3]), m3, res0);
    //    _mm_store_ps(&ret.a[0], res0);
    //    // row 1
    //    __m128 res1 = _mm_mul_ps(_mm_set1_ps(a[4]), m0);
    //    res1 = _mm_fmadd_ps(_mm_set1_ps(a[5]), m1, res1);
    //    res1 = _mm_fmadd_ps(_mm_set1_ps(a[6]), m2, res1);
    //    res1 = _mm_fmadd_ps(_mm_set1_ps(a[7]), m3, res1);
    //    _mm_store_ps(&ret.a[4], res1);
    //    // row 2
    //    __m128 res2 = _mm_mul_ps(_mm_set1_ps(a[8]), m0);
    //    res2 = _mm_fmadd_ps(_mm_set1_ps(a[9]), m1, res2);
    //    res2 = _mm_fmadd_ps(_mm_set1_ps(a[10]), m2, res2);
    //    res2 = _mm_fmadd_ps(_mm_set1_ps(a[11]), m3, res2);
    //    _mm_store_ps(&ret.a[8], res2);
    //    // row 3
    //    __m128 res3 = _mm_mul_ps(_mm_set1_ps(a[12]), m0);
    //    res3 = _mm_fmadd_ps(_mm_set1_ps(a[13]), m1, res3);
    //    res3 = _mm_fmadd_ps(_mm_set1_ps(a[14]), m2, res3);
    //    res3 = _mm_fmadd_ps(_mm_set1_ps(a[15]), m3, res3);
    //    _mm_store_ps(&ret.a[12], res3);
    //    return ret;
    //}




    // Create a perspective projection matrix
    // Input Variables:
    // - fov: Field of view in radians
    // - aspect: Aspect ratio of the viewport
    // - n: Near clipping plane
    // - f: Far clipping plane
    // Returns the perspective matrix
    //static matrix makePerspective(float fov, float aspect, float n, float f) {
    //    matrix m;
    //    m.zero();
    //    float tanHalfFov = std::tan(fov / 2.0f);

    //    m.a[0] = 1.0f / (aspect * tanHalfFov);
    //    m.a[5] = 1.0f / tanHalfFov;
    //    m.a[10] = -f / (f - n);
    //    m.a[11] = -(f * n) / (f - n);
    //    m.a[14] = -1.0f;
    //    return m;
    //}
    // Opt: Set manually
    static matrix makePerspective(float fov, float aspect, float n, float f) {
        matrix m;
        float tanHalfFov = std::tan(fov / 2.0f);
        m.a[0] = 1.0f / (aspect * tanHalfFov); m.a[1] = 0.f; m.a[2] = 0.f; m.a[3] = 0.f;
        m.a[4] = 0.f; m.a[5] = 1.0f / tanHalfFov; m.a[6] = 0.f; m.a[7] = 0.f;
        m.a[8] = 0.f; m.a[9] = 0.f; m.a[10] = -f / (f - n); m.a[11] = -(f * n) / (f - n);
        m.a[12] = 0.f; m.a[13] = 0.f; m.a[14] = -1.f; m.a[15] = 0.f;
        return m;
    }


    // Create a translation matrix
    // Input Variables:
    // - tx, ty, tz: Translation amounts along the X, Y, and Z axes
    // Returns the translation matrix
    //static matrix makeTranslation(float tx, float ty, float tz) {
    //    matrix m;
    //    m.identity();
    //    m.a[3] = tx;
    //    m.a[7] = ty;
    //    m.a[11] = tz;
    //    return m;
    //}
    // Opt: Set manually
    static matrix makeTranslation(float tx, float ty, float tz) {
        matrix m;
        m.a[0] = 1.f; m.a[1] = 0.f; m.a[2] = 0.f; m.a[3] = tx;
        m.a[4] = 0.f; m.a[5] = 1.f; m.a[6] = 0.f; m.a[7] = ty;
        m.a[8] = 0.f; m.a[9] = 0.f; m.a[10] = 1.f; m.a[11] = tz;
        m.a[12] = 0.f; m.a[13] = 0.f; m.a[14] = 0.f; m.a[15] = 1.f;
        return m;
    }

    // Create a rotation matrix around the Z-axis
    // Input Variables:
    // - aRad: Rotation angle in radians
    // Returns the rotation matrix
    //static matrix makeRotateZ(float aRad) {
    //    matrix m;
    //    m.identity();
    //    m.a[0] = std::cos(aRad);
    //    m.a[1] = -std::sin(aRad);
    //    m.a[4] = std::sin(aRad);
    //    m.a[5] = std::cos(aRad);
    //    return m;
    //}
    // Opt:Reduce two calculations by manually assigning values
    static matrix makeRotateZ(float aRad) {
        matrix m;
        float c = std::cos(aRad);
        float s = std::sin(aRad);
        m.a[0] = c;  m.a[1] = -s; m.a[2] = 0.f; m.a[3] = 0.f;
        m.a[4] = s;  m.a[5] = c; m.a[6] = 0.f; m.a[7] = 0.f;
        m.a[8] = 0.f; m.a[9] = 0.f; m.a[10] = 1.f; m.a[11] = 0.f;
        m.a[12] = 0.f; m.a[13] = 0.f; m.a[14] = 0.f; m.a[15] = 1.f;
        return m;
    }

    // Create a rotation matrix around the X-axis
    // Input Variables:
    // - aRad: Rotation angle in radians
    // Returns the rotation matrix
    //static matrix makeRotateX(float aRad) {
    //    matrix m;
    //    m.identity();
    //    m.a[5] = std::cos(aRad);
    //    m.a[6] = -std::sin(aRad);
    //    m.a[9] = std::sin(aRad);
    //    m.a[10] = std::cos(aRad);
    //    return m;
    //}
    // Opt:Reduce two calculations by manually assigning values
    static matrix makeRotateX(float aRad) {
        matrix m;
        float c = std::cos(aRad);
        float s = std::sin(aRad);
        m.a[0] = 1.f; m.a[1] = 0.f; m.a[2] = 0.f; m.a[3] = 0.f;
        m.a[4] = 0.f; m.a[5] = c; m.a[6] = -s; m.a[7] = 0.f;
        m.a[8] = 0.f; m.a[9] = s; m.a[10] = c; m.a[11] = 0.f;
        m.a[12] = 0.f; m.a[13] = 0.f; m.a[14] = 0.f; m.a[15] = 1.f;
        return m;
    }

    // Create a rotation matrix around the Y-axis
    // Input Variables:
    // - aRad: Rotation angle in radians
    // Returns the rotation matrix
    //static matrix makeRotateY(float aRad) {
    //    matrix m;
    //    m.identity();
    //    m.a[0] = std::cos(aRad);
    //    m.a[2] = std::sin(aRad);
    //    m.a[8] = -std::sin(aRad);
    //    m.a[10] = std::cos(aRad);
    //    return m;
    //}
    // Opt:Reduce two calculations by manually assigning values
    static matrix makeRotateY(float aRad) {
        matrix m;
        float c = std::cos(aRad);
        float s = std::sin(aRad);
        m.a[0] = c; m.a[1] = 0.f; m.a[2] = s; m.a[3] = 0.f;
        m.a[4] = 0.f; m.a[5] = 1.f; m.a[6] = 0.f; m.a[7] = 0.f;
        m.a[8] = -s; m.a[9] = 0.f; m.a[10] = c; m.a[11] = 0.f;
        m.a[12] = 0.f; m.a[13] = 0.f; m.a[14] = 0.f; m.a[15] = 1.f;
        return m;
    }

    // Create a composite rotation matrix from X, Y, and Z rotations
    // Input Variables:
    // - x, y, z: Rotation angles in radians around each axis
    // Returns the composite rotation matrix
    static matrix makeRotateXYZ(float x, float y, float z) {
        return matrix::makeRotateX(x) * matrix::makeRotateY(y) * matrix::makeRotateZ(z);
    }

    // Create a scaling matrix
    // Input Variables:
    // - s: Scaling factor
    // Returns the scaling matrix
    //static matrix makeScale(float s) {
    //    matrix m;
    //    s = std::max(s, 0.01f); // Ensure scaling factor is not too small
    //    m.identity();
    //    m.a[0] = s;
    //    m.a[5] = s;
    //    m.a[10] = s;
    //    return m;
    //}
    // Opt: Set manually
    static matrix makeScale(float s) {
        matrix m;
        s = std::max(s, 0.01f); // Ensure scaling factor is not too small
        m.a[0] = s;   m.a[1] = 0.f; m.a[2] = 0.f; m.a[3] = 0.f;
        m.a[4] = 0.f; m.a[5] = s;   m.a[6] = 0.f; m.a[7] = 0.f;
        m.a[8] = 0.f; m.a[9] = 0.f; m.a[10] = s;   m.a[11] = 0.f;
        m.a[12] = 0.f; m.a[13] = 0.f; m.a[14] = 0.f; m.a[15] = 1.f;
        return m;
    }

    // Create an identity matrix
    // Returns an identity matrix
    //static matrix makeIdentity() {
    //    matrix m;
    //    for (int i = 0; i < 4; ++i) {
    //        for (int j = 0; j < 4; ++j) {
    //            m.m[i][j] = (i == j) ? 1.0f : 0.0f;
    //        }
    //    }
    //    return m;
    //}
    static matrix makeIdentity() {
        matrix m;
        m.a[0] = 1.f; m.a[1] = 0.f; m.a[2] = 0.f; m.a[3] = 0.f;
        m.a[4] = 0.f; m.a[5] = 1.f; m.a[6] = 0.f; m.a[7] = 0.f;
        m.a[8] = 0.f; m.a[9] = 0.f; m.a[10] = 1.f; m.a[11] = 0.f;
        m.a[12] = 0.f; m.a[13] = 0.f; m.a[14] = 0.f; m.a[15] = 1.f;
        return m;
    }

private:
    // Set all elements of the matrix to 0
    //void zero() {
    //    for (unsigned int i = 0; i < 16; i++)
    //        a[i] = 0.f;
    //}
    // Opt: SIMD SSE
    void zero() {
        __m128 zero = _mm_setzero_ps();
        _mm_store_ps(&a[0], zero);
        _mm_store_ps(&a[4], zero);
        _mm_store_ps(&a[8], zero);
        _mm_store_ps(&a[12], zero);
    }


    // Set the matrix as an identity matrix
    //void identity() {
    //    for (int i = 0; i < 4; ++i) {
    //        for (int j = 0; j < 4; ++j) {
    //            m[i][j] = (i == j) ? 1.0f : 0.0f;
    //        }
    //    }
    //}
    // Opt: Set manually
    //void identity() {
    //    a[0] = 1.f; a[1] = 0.f; a[2] = 0.f; a[3] = 0.f;
    //    a[4] = 0.f; a[5] = 1.f; a[6] = 0.f; a[7] = 0.f;
    //    a[8] = 0.f; a[9] = 0.f; a[10] = 1.f; a[11] = 0.f;
    //    a[12] = 0.f; a[13] = 0.f; a[14] = 0.f; a[15] = 1.f;
    //}
    // Opt: SIMD
    void identity() {
        _mm_store_ps(&a[0], _mm_set_ps(0.f, 0.f, 0.f, 1.f));  // row 0
        _mm_store_ps(&a[4], _mm_set_ps(0.f, 0.f, 1.f, 0.f));  // row 1
        _mm_store_ps(&a[8], _mm_set_ps(0.f, 1.f, 0.f, 0.f));  // row 2
        _mm_store_ps(&a[12], _mm_set_ps(1.f, 0.f, 0.f, 0.f)); // row 3
    }

};


