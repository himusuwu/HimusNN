#ifndef UTIL_HPP
#define UTIL_HPP

#include "Matrix.hpp"

#include <vector>

Matrix matmul(const Matrix& A, const Matrix& B);
Matrix transpose(const Matrix& A);
Matrix outer(const std::vector<float>& a, const std::vector<float>& b);

#endif
