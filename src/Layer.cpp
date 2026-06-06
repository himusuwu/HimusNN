#include "../include/Layer.hpp"

#include "Matrix.hpp"

#include <cmath>
#include <cstddef>
#include <random>

Layer::Layer(
    int neurons_count, int input_size, ActivationType activation, float leaky_alpha, float elu_alpha, float huber_delta
) :
    W(neurons_count, input_size, 0.0f),
    b(1, neurons_count, 0.0f),
    grad_W(neurons_count, input_size, 0.0f),
    grad_b(1, neurons_count, 0.0f),
    m_W(neurons_count, input_size, 0.0f),
    v_W(neurons_count, input_size, 0.0f),
    m_b(1, neurons_count, 0.0f),
    v_b(1, neurons_count, 0.0f),
    t(0),
    activation(activation),
    leaky_alpha(leaky_alpha),
    elu_alpha(elu_alpha),
    huber_delta(huber_delta)
{
    std::mt19937 rng(std::random_device{}());

    float stddev = std::sqrt(2.0f / static_cast<float>(input_size));

    std::normal_distribution<float> dist(0.0f, stddev);

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

float Layer::activate(float x) const
{
    switch (activation)
    {
        case ActivationType::Sigmoid:
            return 1.0f / (1.0f + exp(-x));
        case ActivationType::Tanh:
            return tanh(x);
        case ActivationType::Relu:
            return (x > 0.0f) ? x : 0.0f;
        case ActivationType::LeakyRelu:
            return (x > 0.0f) ? x : leaky_alpha * x;
        case ActivationType::Elu:
            return (x > 0.0f) ? x : elu_alpha * (exp(x) - 1.0f);
        case ActivationType::Softplus:
            return log(1.0f + exp(x));
        case ActivationType::Linear:
            return x;
    }
    return x;
}

float Layer::activate_derivative(float y) const
{
    switch (activation)
    {
        case ActivationType::Sigmoid:
            return y * (1.0f - y);
        case ActivationType::Tanh:
            return 1.0f - y * y;
        case ActivationType::Relu:
            return (y > 0.0f) ? 1.0f : 0.0f;
        case ActivationType::LeakyRelu:
            return (y > 0.0f) ? 1.0f : leaky_alpha;
        case ActivationType::Elu:
            return (y > 0.0f) ? 1.0f : y + elu_alpha;
        case ActivationType::Softplus:
            return 1.0f / (1.0f + exp(-y));
        case ActivationType::Linear:
            return 1.0f;
    }
    return 1.0f;
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

    for (size_t i = 0; i < Z.data.size(); i++)
    {
        Z.data[i] = activate(Z.data[i]);
    }

    last_output = Z;

    return last_output;
}

Matrix Layer::backward(const Matrix& dA, float learning_rate, float beta1, float beta2, size_t batch_size)
{
    Matrix dZ(dA.rows, dA.cols);

    for (size_t i = 0; i < dZ.data.size(); i++)
    {
        dZ.data[i] = dA.data[i] * activate_derivative(last_output.data[i]);
    }

    grad_W = dZ.transposed().matmul(last_input).scaled(1.0f / static_cast<float>(batch_size));
    grad_b = dZ.mean_axis0();

    t++;

    m_W = m_W.scaled(beta1).add(grad_W.scaled(1.0f - beta1));
    v_W = v_W.scaled(beta2).add(grad_W.hadamard(grad_W).scaled(1.0f - beta2));

    Matrix m_W_hat = m_W.scaled(1.0f / (1.0f - std::pow(beta1, t)));
    Matrix v_W_hat = v_W.scaled(1.0f / (1.0f - std::pow(beta2, t)));

    float epsilon = 1e-7f;

    for (size_t i = 0; i < W.data.size(); i++)
    {
        W.data[i] += (-learning_rate) * (m_W_hat.data[i] / (std::sqrt(v_W_hat.data[i]) + epsilon));
    }

    m_b = m_b.scaled(beta1).add(grad_b.scaled(1.0f - beta1));
    v_b = v_b.scaled(beta2).add(grad_b.hadamard(grad_b).scaled(1.0f - beta2));

    Matrix m_b_hat = m_b.scaled(1.0f / (1.0f - std::pow(beta1, t)));
    Matrix v_b_hat = v_b.scaled(1.0f / (1.0f - std::pow(beta2, t)));

    for (size_t i = 0; i < b.data.size(); i++)
    {
        b.data[i] += (-learning_rate) * (m_b_hat.data[i] / (std::sqrt(v_b_hat.data[i]) + epsilon));
    }

    Matrix dA_prev = dZ.matmul(W);

    return dA_prev;
}

Matrix Layer::backward_from_dZ(const Matrix& dZ, float learning_rate, float beta1, float beta2, size_t batch_size)
{
    grad_W = dZ.transposed().matmul(last_input).scaled(1.0f / static_cast<float>(batch_size));
    grad_b = dZ.mean_axis0();

    t++;

    m_W = m_W.scaled(beta1).add(grad_W.scaled(1.0f - beta1));
    v_W = v_W.scaled(beta2).add(grad_W.hadamard(grad_W).scaled(1.0f - beta2));

    Matrix m_W_hat = m_W.scaled(1.0f / (1.0f - std::pow(beta1, t)));
    Matrix v_W_hat = v_W.scaled(1.0f / (1.0f - std::pow(beta2, t)));

    float epsilon = 1e-7f;

    for (size_t i = 0; i < W.data.size(); i++)
    {
        W.data[i] += (-learning_rate) * (m_W_hat.data[i] / (std::sqrt(v_W_hat.data[i]) + epsilon));
    }

    m_b = m_b.scaled(beta1).add(grad_b.scaled(1.0f - beta1));
    v_b = v_b.scaled(beta2).add(grad_b.hadamard(grad_b).scaled(1.0 - beta2));

    Matrix m_b_hat = m_b.scaled(1.0f / (1.0f - std::pow(beta1, t)));
    Matrix v_b_hat = v_b.scaled(1.0f / (1.0f - std::pow(beta2, t)));

    for (size_t i = 0; i < b.data.size(); i++)
    {
        b.data[i] += (-learning_rate) * (m_b_hat.data[i] / (std::sqrt(v_b_hat.data[i]) + epsilon));
    }

    Matrix dA_prev = dZ.matmul(W);

    return dA_prev;
}
