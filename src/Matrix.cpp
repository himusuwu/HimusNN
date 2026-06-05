#include "Matrix.hpp"

#include "util.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

Matrix::Matrix()
{
    rows = 0;
    cols = 0;
}

Matrix::Matrix(size_t rows, size_t cols, double fill) : rows(rows), cols(cols)
{
    data.resize(rows * cols, fill);
}

Matrix::Matrix(size_t rows, size_t cols, const std::vector<double>& init_data) : rows(rows), cols(cols)
{
    if (init_data.size() == rows * cols)
    {
        data = init_data;
    }
    else
    {
        throw std::invalid_argument("Init data size is wrong.");
    }
}

Matrix Matrix::zeros(size_t rows, size_t cols)
{
    return (Matrix(rows, cols, 0.0));
}

Matrix Matrix::ones(size_t rows, size_t cols)
{
    return (Matrix(rows, cols, 1.0));
}

Matrix Matrix::identity(size_t n)
{
    Matrix m = Matrix(n, n, 0.0);

    for (size_t i = 0; i < n; i++)
    {
        m(i, i) = 1.0;
    }
    return m;
}

Matrix Matrix::from_vector(const std::vector<double>& v, bool as_row)
{
    if (as_row)
    {
        return (Matrix(1, v.size(), v));
    }
    else
    {
        return (Matrix(v.size(), 1, v));
    }
}

void Matrix::check_index(size_t r, size_t c) const
{
    if (r >= rows || c >= cols)
    {
        throw std::out_of_range("Index out of range.");
    }
}

void Matrix::assert_dims(size_t r, size_t c) const
{
    if (r != rows || c != cols)
    {
        throw std::invalid_argument("Invalid argument was given.");
    }
}

double& Matrix::operator()(size_t r, size_t c)
{
    check_index(r, c);

    return data[r * cols + c];
}

double Matrix::operator()(size_t r, size_t c) const
{
    check_index(r, c);

    return data[r * cols + c];
}

double* Matrix::data_ptr()
{
    return data.data();
}

const double* Matrix::data_ptr() const
{
    return data.data();
}

size_t Matrix::size() const
{
    return rows * cols;
}

void Matrix::fill(double value)
{
    std::fill(data.begin(), data.end(), value);
}

void Matrix::scale_inplace(double s)
{
    for (double& d : data)
    {
        d *= s;
    }
}

void Matrix::add_inplace(const Matrix& other)
{
    assert_dims(other.rows, other.cols);

    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] += other.data[i];
    }
}

void Matrix::sub_inplace(const Matrix& other)
{
    assert_dims(other.rows, other.cols);

    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] -= other.data[i];
    }
}

void Matrix::hadamard_inplace(const Matrix& other)
{
    assert_dims(other.rows, other.cols);

    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] *= other.data[i];
    }
}

Matrix Matrix::transposed() const
{
    Matrix out(cols, rows);

    for (size_t c = 0; c < cols; c++)
    {
        for (size_t r = 0; r < rows; r++)
        {
            out(c, r) = (*this)(r, c);
        }
    }

    return out;
}

Matrix Matrix::add(const Matrix& other) const
{
    assert_dims(other.rows, other.cols);

    Matrix out(rows, cols);

    for (size_t i = 0; i < data.size(); i++)
    {
        out.data[i] = data[i] + other.data[i];
    }

    return out;
}

Matrix Matrix::sub(const Matrix& other) const
{
    assert_dims(other.rows, other.cols);

    Matrix out(rows, cols);

    for (size_t i = 0; i < data.size(); i++)
    {
        out.data[i] = data[i] - other.data[i];
    }

    return out;
}

Matrix Matrix::hadamard(const Matrix& other) const
{
    assert_dims(other.rows, other.cols);

    Matrix out(rows, cols);

    for (size_t i = 0; i < data.size(); i++)
    {
        out.data[i] = data[i] * other.data[i];
    }

    return out;
}

Matrix Matrix::scaled(double s) const
{
    Matrix out(rows, cols);

    for (size_t i = 0; i < data.size(); i++)
    {
        out.data[i] = data[i] * s;
    }

    return out;
}

Matrix Matrix::matmul(const Matrix& other) const
{
    if (cols != other.rows)
    {
        throw std::invalid_argument("Invalid argument for matmul.");
    }

    Matrix out(rows, other.cols, 0.0);

    for (size_t r = 0; r < rows; r++)
    {
        for (size_t k = 0; k < cols; k++)
        {
            for (size_t c = 0; c < other.cols; c++)
            {
                out(r, c) += (*this)(r, k) * other(k, c);
            }
        }
    }

    return out;
}

std::vector<double> Matrix::col(size_t c) const
{
    check_index(0, c);

    std::vector<double> out;
    out.resize(rows);

    for (size_t r = 0; r < rows; r++)
    {
        out[r] = (*this)(r, c);
    }

    return out;
}

std::vector<double> Matrix::row(size_t r) const
{
    check_index(r, 0);

    std::vector<double> out;
    out.resize(cols);

    for (size_t c = 0; c < cols; c++)
    {
        out[c] = (*this)(r, c);
    }

    return out;
}

void Matrix::set_row(size_t r, const std::vector<double>& v)
{
    check_index(r, 0);

    if (v.size() != cols)
    {
        throw std::invalid_argument("Invalid argument for set_row.");
    }

    for (size_t c = 0; c < cols; c++)
    {
        (*this)(r, c) = v[c];
    }
}

void Matrix::set_col(size_t c, const std::vector<double>& v)
{
    check_index(0, c);

    if (v.size() != rows)
    {
        throw std::invalid_argument("Invalid argument for set_col.");
    }

    for (size_t r = 0; r < rows; r++)
    {
        (*this)(r, c) = v[r];
    }
}

Matrix Matrix::sum_axis0() const
{
    Matrix out(1, cols, 0.0);

    for (size_t r = 0; r < rows; r++)
    {
        for (size_t c = 0; c < cols; c++)
        {
            out(0, c) += (*this)(r, c);
        }
    }

    return out;
}

Matrix Matrix::sum_axis1() const
{
    Matrix out(rows, 1, 0.0);

    for (size_t r = 0; r < rows; r++)
    {
        for (size_t c = 0; c < cols; c++)
        {
            out(r, 0) += (*this)(r, c);
        }
    }

    return out;
}

Matrix Matrix::mean_axis0() const
{
    if (rows == 0)
    {
        throw std::invalid_argument("mean_axis0 on empty matrix.");
    }

    return (sum_axis0().scaled(1.0 / static_cast<double>(rows)));
}

Matrix Matrix::mean_axis1() const
{
    if (cols == 0)
    {
        throw std::invalid_argument("mean_axis1 on empty matrix.");
    }

    return (sum_axis1().scaled(1.0 / static_cast<double>(cols)));
}

Matrix Matrix::apply(std::function<double(double)> f) const
{
    Matrix out(rows, cols);

    for (size_t i = 0; i < data.size(); i++)
    {
        out.data[i] = f(data[i]);
    }

    return out;
}

void Matrix::apply_inplace(std::function<double(double)> f)
{
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] = f(data[i]);
    }
}

Matrix Matrix::outer(const std::vector<double>& a, const std::vector<double>& b)
{
    Matrix out(a.size(), b.size());

    for (size_t i = 0; i < a.size(); i++)
    {
        for (size_t j = 0; j < b.size(); j++)
        {
            out(i, j) = a[i] * b[j];
        }
    }

    return out;
}

Matrix outer(const std::vector<double>& a, const std::vector<double>& b)
{
    return (Matrix::outer(a, b));
}

Matrix transpose(const Matrix& A)
{
    return (A.transposed());
}

Matrix matmul(const Matrix& A, const Matrix& B)
{
    return A.matmul(B);
}
