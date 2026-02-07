#pragma once
#include "matrix_opt.h"
#include "colour.h"
//using matrix = matrixT<Matrix_Mul_Scalar, Matrix_Trans_Intial>;
// best
using matrix = matrixT<Matrix_Mul_SIMD_DP, Matrix_Trans_Intial>;
//using matrix = matrixT<Matrix_Mul_SIMD_DP2, Matrix_Trans_Intial>;
//using matrix = matrixT<Matrix_Mul_SIMD, Matrix_Trans_Intial>;
//using matrix = matrixT<Matrix_Mul_SIMD, Matrix_Trans_Manually>;
//using matrix = matrixT<Matrix_Mul_SIMD_DP, Matrix_Trans_Manually>;
//using matrix = matrixT<Matrix_Mul_SIMD, Matrix_Trans_SIMD>;

//using colour = colour_rgb;
using colour = colour_opt;