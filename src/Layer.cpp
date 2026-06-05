#include "../include/Layer.hpp"

#include "Matrix.hpp"

#include <cmath>
#include <cstddef>
#include <random>

Layer::Layer(
    int neurons_count,
    int input_size,
    ActivationType activation,
    double leaky_alpha,
    double elu_alpha,
    double huber_delta
) :
    W(neurons_count, input_size, 0.0),
    b(1, neurons_count, 0.0),
    grad_W(neurons_count, input_size, 0.0),
    grad_b(1, neurons_count, 0.0),
    velocity_W(neurons_count, input_size, 0.0),
    velocity_b(1, neurons_count, 0.0),
    activation(activation),
    leaky_alpha(leaky_alpha),
    elu_alpha(elu_alpha),
    huber_delta(huber_delta)
{
    std::mt19937 rng(std::random_device{}());

    double stddev = std::sqrt(2.0 / static_cast<double>(input_size));

    std::normal_distribution<double> dist(0.0, stddev);

    for (size_t r = 0; r < W.rows; r++)
    {
        for (size_t c = 0; c < W.cols; c++)
        {
            W(r, c) = dist(rng);
        }
    }

    for (size_t c = 0; c < b.cols; c++)
    {
        b(0, c) = dist(rng);
    }
}

double Layer::activate(double x) const
{
    switch (activation)
    {
        case ActivationType::Sigmoid:
            return 1.0 / (1.0 + exp(-x));
        case ActivationType::Tanh:
            return tanh(x);
        case ActivationType::Relu:
            return (x > 0.0) ? x : 0.0;
        case ActivationType::LeakyRelu:
            return (x > 0.0) ? x : leaky_alpha * x;
        case ActivationType::Elu:
            return (x > 0.0) ? x : elu_alpha * (exp(x) - 1.0);
        case ActivationType::Softplus:
            return log(1.0 + exp(x));
        case ActivationType::Linear:
            return x;
    }
    return x;
}

double Layer::activate_derivative(double y) const
{
    switch (activation)
    {
        case ActivationType::Sigmoid:
            return y * (1.0 - y);
        case ActivationType::Tanh:
            return 1.0 - y * y;
        case ActivationType::Relu:
            return (y > 0.0) ? 1.0 : 0.0;
        case ActivationType::LeakyRelu:
            return (y > 0.0) ? 1.0 : leaky_alpha;
        case ActivationType::Elu:
            return (y > 0.0) ? 1.0 : y + elu_alpha;
        case ActivationType::Softplus:
            return 1.0 / (1.0 + exp(-y));
        case ActivationType::Linear:
            return 1.0;
    }
    return 1.0;
}

Matrix Layer::forward(const Matrix& X)
{
    last_input = X;                      // batch x in

    Matrix Z = X.matmul(W.transposed()); // (batch x in) * (in x out) = batch x out

    // add bias row-wise
    for (size_t r = 0; r < Z.rows; r++)
    {
        for (size_t c = 0; c < Z.cols; c++)
        {
            Z(r, c) += b(0, c);
        }
    }

    last_output = Z.apply([this](double v) { return activate(v); });
    return last_output;
}

Matrix Layer::backward(const Matrix& dA, double learning_rate, double momentum, size_t batch_size)
{
    Matrix dZ = dA.hadamard(last_output.apply([this](double y) { return activate_derivative(y); }));

    grad_W = dZ.transposed().matmul(last_input).scaled(1.0 / static_cast<double>(batch_size));
    grad_b = dZ.mean_axis0();

    velocity_W = velocity_W.scaled(momentum).add(grad_W);
    velocity_b = velocity_b.scaled(momentum).add(grad_b);

    W.add_inplace(velocity_W.scaled(learning_rate));
    b.add_inplace(velocity_b.scaled(learning_rate));

    Matrix dA_prev = dZ.matmul(W);

    return dA_prev;
}

Matrix Layer::backward_from_dZ(const Matrix& dZ, double learning_rate, double momentum, size_t batch_size)
{
    grad_W = dZ.transposed().matmul(last_input).scaled(1.0 / static_cast<double>(batch_size));
    grad_b = dZ.mean_axis0();

    velocity_W = velocity_W.scaled(momentum).add(grad_W);
    velocity_b = velocity_b.scaled(momentum).add(grad_b);

    W.add_inplace(velocity_W.scaled(learning_rate));
    b.add_inplace(velocity_b.scaled(learning_rate));

    Matrix dA_prev = dZ.matmul(W);

    return dA_prev;
}
