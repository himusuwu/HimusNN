#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <span>
#include <vector>

#ifdef __APPLE__
    #include <dispatch/dispatch.h>
#endif

struct Matrix
{
    size_t rows;
    size_t cols;
    std::vector<float> data;

    // construstors, copy, move
    Matrix();
    Matrix(size_t rows, size_t cols, float fill = 0.0f);
    Matrix(size_t rows, size_t cols, const std::vector<float>& init_data); // init_data.size() == rows * cols
    Matrix(size_t rows, size_t cols, std::span<const float> data_span);
    Matrix(const Matrix&) = default;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&&) noexcept = default;

    // static factory
    [[nodiscard]] static Matrix zeros(size_t rows, size_t cols);
    [[nodiscard]] static Matrix ones(size_t rows, size_t cols);
    [[nodiscard]] static Matrix identity(size_t n);
    [[nodiscard]] static Matrix
    from_vector(const std::vector<float>& v, bool as_row = true); // as_row -> 1 x n, else n x 1

    // access, info
    [[nodiscard]] float& operator()(size_t r, size_t c);
    [[nodiscard]] float operator()(size_t r, size_t c) const;
    [[nodiscard]] float* data_ptr();
    [[nodiscard]] const float* data_ptr() const;
    [[nodiscard]] size_t size() const; // rows * cols

    // in-place operations
    void fill(float value);
    void scale_inplace(float s);                // all elements * s
    void add_inplace(const Matrix& other);      // dims must match
    void sub_inplace(const Matrix& other);
    void hadamard_inplace(const Matrix& other); // elementwise multiply

    // new matrix as return
    [[nodiscard]] Matrix transposed() const;
    [[nodiscard]] Matrix add(const Matrix& other) const;
    [[nodiscard]] Matrix sub(const Matrix& other) const;
    [[nodiscard]] Matrix hadamard(const Matrix& other) const;
    [[nodiscard]] Matrix scaled(float s) const;
    [[nodiscard]] Matrix matmul(const Matrix& other, bool transA = false, bool transB = false) const; // this * other

    // rows / cols
    [[nodiscard]] std::vector<float> row(size_t r) const;
    [[nodiscard]] std::vector<float> col(size_t c) const;
    void set_row(size_t r, const std::vector<float>& v);
    void set_col(size_t c, const std::vector<float>& v);

    // reductions
    [[nodiscard]] Matrix sum_axis0() const; // returns 1 x cols
    [[nodiscard]] Matrix sum_axis1() const; // returns rows x 1
    [[nodiscard]] Matrix mean_axis0() const;
    [[nodiscard]] Matrix mean_axis1() const;

    // apply
    [[nodiscard]] Matrix apply(float (*f)(float)) const; // returns new matrix
    void apply_inplace(float (*f)(float));               // modifies in place

    template <class F> [[nodiscard]] Matrix apply_unary(F f) const
    {
        Matrix out(rows, cols);

#ifdef __APPLE__
        float* out_ptr = out.data_ptr();
        const float* in_ptr = this->data_ptr();

        dispatch_apply(data.size(), dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t i) {
            out_ptr[i] = f(in_ptr[i]);
        });
#else
        for (size_t i = 0; i < data.size(); i++)
        {
            out.data[i] = f(data[i]);
        }
#endif

        return out;
    }

    // helpers
    [[nodiscard]] static Matrix outer(const std::vector<float>& a, const std::vector<float>& b); // outer product
    void assert_dims(size_t r, size_t c) const; // throws std::invalid_argument when mismatch
    void check_index(size_t r, size_t c) const; // throws std::out_of_range
};

#endif
