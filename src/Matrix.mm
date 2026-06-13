#include <dispatch/dispatch.h>
#include <vecLib/cblas_new.h>
#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>

    // Metal
    #import <Metal/Metal.h>
    #import <MetalPerformanceShaders/MetalPerformanceShaders.h>

static id<MTLDevice> metalDevice = nil;
static id<MTLCommandQueue> commandQueue = nil;

void initMetal()
{
    if (!metalDevice)
    {
        metalDevice = MTLCreateSystemDefaultDevice();
        commandQueue = [metalDevice newCommandQueue];
    }
}
#endif

#include "Matrix.hpp"
#include "util.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

Matrix::Matrix()
{
    rows = 0;
    cols = 0;
}

Matrix::Matrix(size_t rows, size_t cols, float fill) : rows(rows), cols(cols)
{
    data.resize(rows * cols, fill);
}

Matrix::Matrix(size_t rows, size_t cols, const std::vector<float>& init_data) : rows(rows), cols(cols)
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

Matrix::Matrix(size_t rows, size_t cols, std::span<const float> data_span) : rows(rows), cols(cols)
{
    if (data_span.size() != rows * cols) [[unlikely]]
    {
        throw std::invalid_argument("Span size does not match matrix dimensions.");
    }

    data.assign(data_span.begin(), data_span.end());
}

Matrix Matrix::zeros(size_t rows, size_t cols)
{
    return (Matrix(rows, cols, 0.0f));
}

Matrix Matrix::ones(size_t rows, size_t cols)
{
    return (Matrix(rows, cols, 1.0f));
}

Matrix Matrix::identity(size_t n)
{
    Matrix m = Matrix(n, n, 0.0f);

    for (size_t i = 0; i < n; i++)
    {
        m(i, i) = 1.0f;
    }
    return m;
}

Matrix Matrix::from_vector(const std::vector<float>& v, bool as_row)
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
    if (r >= rows || c >= cols) [[unlikely]]
    {
        throw std::out_of_range("Index out of range.");
    }
}

void Matrix::assert_dims(size_t r, size_t c) const
{
    if (r != rows || c != cols) [[unlikely]]
    {
        throw std::invalid_argument("Invalid argument was given.");
    }
}

float& Matrix::operator()(size_t r, size_t c)
{
    check_index(r, c);

    return data[r * cols + c];
}

float Matrix::operator()(size_t r, size_t c) const
{
    check_index(r, c);

    return data[r * cols + c];
}

float* Matrix::data_ptr()
{
    return data.data();
}

const float* Matrix::data_ptr() const
{
    return data.data();
}

size_t Matrix::size() const
{
    return rows * cols;
}

void Matrix::fill(float value)
{
    std::fill(data.begin(), data.end(), value);
}

void Matrix::scale_inplace(float s)
{
#ifdef __APPLE__
    vDSP_vsmul(data.data(), 1, &s, data.data(), 1, data.size());
#else
    for (float& d : data)
    {
        d *= s;
    }
#endif
}

void Matrix::add_inplace(const Matrix& other)
{
    assert_dims(other.rows, other.cols);

#ifdef __APPLE__
    vDSP_vadd(data.data(), 1, other.data.data(), 1, data.data(), 1, data.size());
#else
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] += other.data[i];
    }
#endif
}

void Matrix::sub_inplace(const Matrix& other)
{
    assert_dims(other.rows, other.cols);

#ifdef __APPLE__
    vDSP_vsub(other.data.data(), 1, data.data(), 1, data.data(), 1, data.size());
#else
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] -= other.data[i];
    }
#endif
}

void Matrix::hadamard_inplace(const Matrix& other)
{
    assert_dims(other.rows, other.cols);

#ifdef __APPLE__
    vDSP_vmul(data.data(), 1, other.data.data(), 1, data.data(), 1, data.size());
#else
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] *= other.data[i];
    }
#endif
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

#ifdef __APPLE__
    vDSP_vadd(data.data(), 1, other.data.data(), 1, out.data.data(), 1, data.size());
#else
    for (size_t i = 0; i < data.size(); i++)
    {
        out.data[i] = data[i] + other.data[i];
    }
#endif

    return out;
}

Matrix Matrix::sub(const Matrix& other) const
{
    assert_dims(other.rows, other.cols);

    Matrix out(rows, cols);

#ifdef __APPLE__
    vDSP_vsub(other.data.data(), 1, data.data(), 1, out.data.data(), 1, data.size());
#else
    for (size_t i = 0; i < data.size(); i++)
    {
        out.data[i] = data[i] - other.data[i];
    }
#endif

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

Matrix Matrix::scaled(float scalar) const
{
    Matrix out(rows, cols);

#ifdef __APPLE__
    vDSP_vsmul(data.data(), 1, &scalar, out.data.data(), 1, data.size());
#else
    for (size_t i = 0; i < data.size(); i++)
    {
        out.data[i] = data[i] * scalar;
    }
#endif

    return out;
}

Matrix Matrix::matmul(const Matrix& other, bool transA, bool transB) const
{
    size_t realRowsA = transA ? cols : rows;
    size_t realColsA = transA ? rows : cols;
    size_t realRowsB = transB ? other.cols : other.rows;
    size_t realColsB = transB ? other.rows : other.cols;

    if (realColsA != realRowsB) [[unlikely]]
    {
        throw std::invalid_argument("Invalid argument for matmul (dimension mismatch).");
    }

    Matrix out(realRowsA, realColsB, 0.0f);

#ifdef __APPLE__
    size_t complexity = rows * cols * other.cols;

    if (complexity < 5000000)
    {
        static bool printedCPU = false;
        if (!printedCPU)
        {
            std::cout << "[DEBUG] Using Apple Accelerate (CPU/AMX) for matrix.\n";
            printedCPU = true;
        }

        int lda = transA ? static_cast<int>(realRowsA) : static_cast<int>(realColsA);
        int ldb = transB ? static_cast<int>(realRowsB) : static_cast<int>(realColsB);

        cblas_sgemm(
            CblasRowMajor,
            transA ? CblasTrans : CblasNoTrans,
            transB ? CblasTrans : CblasNoTrans,
            static_cast<int>(realRowsA),
            static_cast<int>(realColsB),
            static_cast<int>(realColsA),
            1.0f,
            data.data(),
            lda,
            other.data.data(),
            ldb,
            0.0f,
            out.data.data(),
            static_cast<int>(out.cols)
        );
    }
    else
    {
        // clang-format off

		static bool printedGPU = false;
        if (!printedGPU)
        {
            std::cout << "[DEBUG] Using Apple Metal (GPU) for matrix.\n";
            printedGPU = true;
        }

        initMetal();

        id<MTLBuffer> bufA = [
            metalDevice 
            newBufferWithBytesNoCopy:(void*)data.data()
            length:data.size() * sizeof(float)
            options:MTLResourceStorageModeShared
            deallocator:nil
        ];

        id<MTLBuffer> bufB = [
            metalDevice
            newBufferWithBytesNoCopy:(void*)other.data.data()
            length:other.data.size() * sizeof(float)
            options:MTLResourceStorageModeShared
            deallocator:nil
        ];

        id<MTLBuffer> bufC = [
            metalDevice newBufferWithBytesNoCopy:(void*)out.data.data()
            length:out.data.size() * sizeof(float)
            options:MTLResourceStorageModeShared
            deallocator:nil
        ];

        MPSMatrixDescriptor *descA = [
            MPSMatrixDescriptor 
            matrixDescriptorWithRows:rows 
            columns:cols 
            rowBytes:cols * sizeof(float) 
            dataType:MPSDataTypeFloat32
        ];

        MPSMatrixDescriptor *descB = [
            MPSMatrixDescriptor 
            matrixDescriptorWithRows:other.rows 
            columns:other.cols 
            rowBytes:other.cols * sizeof(float) 
            dataType:MPSDataTypeFloat32
        ];

        MPSMatrixDescriptor *descC = [
            MPSMatrixDescriptor 
            matrixDescriptorWithRows:realRowsA 
            columns:realColsB 
            rowBytes:realColsB * sizeof(float) 
            dataType:MPSDataTypeFloat32
        ];

        MPSMatrix *matA = [
            [MPSMatrix alloc]
            initWithBuffer:bufA
            descriptor:descA
        ];

        MPSMatrix *matB = [
            [MPSMatrix alloc]
            initWithBuffer:bufB
            descriptor:descB
        ];

        MPSMatrix *matC = [
            [MPSMatrix alloc]
            initWithBuffer:bufC
            descriptor:descC
        ];

        MPSMatrixMultiplication *mul = [
            [MPSMatrixMultiplication alloc]
            initWithDevice:metalDevice
            transposeLeft:(transA ? YES : NO)
            transposeRight:(transB ? YES : NO)
            resultRows:realRowsA
            resultColumns:realColsB
            interiorColumns:realColsA
            alpha:1.0f
            beta:0.0f
        ];

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [
            mul encodeToCommandBuffer:commandBuffer
            leftMatrix:matA
            rightMatrix:matB
            resultMatrix:matC
        ];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // clang-format on
    }
#else
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
#endif

    return out;
}

std::vector<float> Matrix::col(size_t c) const
{
    check_index(0, c);

    std::vector<float> out;
    out.resize(rows);

    for (size_t r = 0; r < rows; r++)
    {
        out[r] = (*this)(r, c);
    }

    return out;
}

std::vector<float> Matrix::row(size_t r) const
{
    check_index(r, 0);

    std::vector<float> out;
    out.resize(cols);

    for (size_t c = 0; c < cols; c++)
    {
        out[c] = (*this)(r, c);
    }

    return out;
}

void Matrix::set_row(size_t r, const std::vector<float>& v)
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

void Matrix::set_col(size_t c, const std::vector<float>& v)
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
    Matrix out(1, cols, 0.0f);

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
    Matrix out(rows, 1, 0.0f);

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
    if (rows == 0) [[unlikely]]
    {
        throw std::invalid_argument("mean_axis0 on empty matrix.");
    }

    return (sum_axis0().scaled(1.0f / static_cast<float>(rows)));
}

Matrix Matrix::mean_axis1() const
{
    if (cols == 0) [[unlikely]]
    {
        throw std::invalid_argument("mean_axis1 on empty matrix.");
    }

    return (sum_axis1().scaled(1.0f / static_cast<float>(cols)));
}

Matrix Matrix::apply(float (*f)(float)) const
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

void Matrix::apply_inplace(float (*f)(float))
{
#ifdef __APPLE__
    float* ptr = this->data_ptr();

    dispatch_apply(data.size(), dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t i) {
        ptr[i] = f(ptr[i]);
    });
#else
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] = f(data[i]);
    }
#endif
}

Matrix Matrix::outer(const std::vector<float>& a, const std::vector<float>& b)
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

Matrix outer(const std::vector<float>& a, const std::vector<float>& b)
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
