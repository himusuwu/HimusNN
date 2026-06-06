#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <functional>
#include <vector>

struct Matrix
{
    size_t rows;
    size_t cols;
    std::vector<float> data;

    // construstors, copy, move
    Matrix();
    Matrix(size_t rows, size_t cols, float fill = 0.0f);
    Matrix(size_t rows, size_t cols, const std::vector<float>& init_data); // init_data.size() == rows * cols
    Matrix(const Matrix&) = default;
    Matrix(Matrix&&) = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&&) = default;

    // static factory
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix identity(size_t n);
    static Matrix from_vector(const std::vector<float>& v, bool as_row = true); // as_row -> 1 x n, else n x 1

    // access, info
    float& operator()(size_t r, size_t c);
    float operator()(size_t r, size_t c) const;
    float* data_ptr();
    const float* data_ptr() const;
    size_t size() const; // rows * cols

    // in-place operations
    void fill(float value);
    void scale_inplace(float s);                // all elements * s
    void add_inplace(const Matrix& other);      // dims must match
    void sub_inplace(const Matrix& other);
    void hadamard_inplace(const Matrix& other); // elementwise multiply

    // new matrix as return
    Matrix transposed() const;
    Matrix add(const Matrix& other) const;
    Matrix sub(const Matrix& other) const;
    Matrix hadamard(const Matrix& other) const;
    Matrix scaled(float s) const;
    Matrix matmul(const Matrix& other) const; // this * other

    // rows / cols
    std::vector<float> row(size_t r) const;
    std::vector<float> col(size_t c) const;
    void set_row(size_t r, const std::vector<float>& v);
    void set_col(size_t c, const std::vector<float>& v);

    // reductions
    Matrix sum_axis0() const; // returns 1 x cols
    Matrix sum_axis1() const; // returns rows x 1
    Matrix mean_axis0() const;
    Matrix mean_axis1() const;

    // apply
    Matrix apply(float (*f)(float)) const; // returns new matrix
    void apply_inplace(float (*f)(float)); // modifies in place

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
    static Matrix outer(const std::vector<float>& a, const std::vector<float>& b); // outer product
    void assert_dims(size_t r, size_t c) const; // throws std::invalid_argument when mismatch
    void check_index(size_t r, size_t c) const; // throws std::out_of_range
};

#endif
