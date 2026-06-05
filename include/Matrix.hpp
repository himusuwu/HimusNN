#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <functional>
#include <vector>

struct Matrix
{
    size_t rows;
    size_t cols;
    std::vector<double> data;

    // construstors, copy, move
    Matrix();
    Matrix(size_t rows, size_t cols, double fill = 0.0);
    Matrix(size_t rows, size_t cols, const std::vector<double>& init_data); // init_data.size() == rows * cols
    Matrix(const Matrix&) = default;
    Matrix(Matrix&&) = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&&) = default;

    // static factory
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix identity(size_t n);
    static Matrix from_vector(const std::vector<double>& v, bool as_row = true); // as_row -> 1 x n, else n x 1

    // access, info
    double& operator()(size_t r, size_t c);
    double operator()(size_t r, size_t c) const;
    double* data_ptr();
    const double* data_ptr() const;
    size_t size() const; // rows * cols

    // in-place operations
    void fill(double value);
    void scale_inplace(double s);               // all elements * s
    void add_inplace(const Matrix& other);      // dims must match
    void sub_inplace(const Matrix& other);
    void hadamard_inplace(const Matrix& other); // elementwise multiply

    // new matrix as return
    Matrix transposed() const;
    Matrix add(const Matrix& other) const;
    Matrix sub(const Matrix& other) const;
    Matrix hadamard(const Matrix& other) const;
    Matrix scaled(double s) const;
    Matrix matmul(const Matrix& other) const; // this * other

    // rows / cols
    std::vector<double> row(size_t r) const;
    std::vector<double> col(size_t c) const;
    void set_row(size_t r, const std::vector<double>& v);
    void set_col(size_t c, const std::vector<double>& v);

    // reductions
    Matrix sum_axis0() const; // returns 1 x cols
    Matrix sum_axis1() const; // returns rows x 1
    Matrix mean_axis0() const;
    Matrix mean_axis1() const;

    // apply
    Matrix apply(std::function<double(double)> f) const; // returns new matrix
    void apply_inplace(std::function<double(double)> f); // modifies in place

    template <class F> Matrix apply_unary(F f) const
    {
        Matrix out(rows, cols);

        for (size_t i = 0; i < data.size(); i++)
        {
            out.data[i] = f(data[i]);
        }

        return out;
    }

    // helpers
    static Matrix outer(const std::vector<double>& a, const std::vector<double>& b); // outer product
    void assert_dims(size_t r, size_t c) const; // throws std::invalid_argument when mismatch
    void check_index(size_t r, size_t c) const; // throws std::out_of_range
};

#endif
