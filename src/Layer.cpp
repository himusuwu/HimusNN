#include "../include/Layer.hpp"

#include "Config.hpp"
#include "Matrix.hpp"
#include "util.hpp"

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
        case ActivationType::Softmax:
            return x; // done in forward function
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
        case ActivationType::Softmax:
            return 1.0f; // done in Network.cpp
    }
    return 1.0f;
}

Matrix Layer::forward(const Matrix& X)
{
    last_input = X;                      // batch x in

    Matrix Z = X.matmul(W, false, true); // (batch x in) * (in x out) = batch x out

    size_t rows = Z.rows;
    size_t cols = Z.cols;

    if (activation == ActivationType::Softmax)
    {
        paraller_for(
            size_t(0),
            rows,
            [&](size_t r)
            {
                float max_val = -1e30f;

                for (size_t c = 0; c < cols; c++)
                {
                    Z(r, c) += b(0, c);

                    if (Z(r, c) > max_val)
                    {
                        max_val = Z(r, c);
                    }
                }

                float sum = 0.0f;

                for (size_t c = 0; c < Z.cols; c++)
                {
                    Z(r, c) = exp(Z(r, c) - max_val); // e^(z - max)
                    sum += Z(r, c);
                }

                for (size_t c = 0; c < Z.cols; c++)
                {
                    Z(r, c) /= (sum + 1e-15f); // Normalization
                }
            }
        );
    }
    else
    {
        paraller_for(
            size_t(0),
            rows,
            [&](size_t r)
            {
                float* row_ptr = &Z.data[r * cols];
                const float* bias_ptr = b.data.data();

                for (size_t c = 0; c < cols; c++)
                {
                    row_ptr[c] = activate(row_ptr[c] + bias_ptr[c]);
                }
            }
        );
    }

    last_output = Z;

    return last_output;
}

Matrix Layer::backward(const Matrix& dA, float learning_rate, float beta1, float beta2, size_t batch_size)
{
    Matrix dZ(dA.rows, dA.cols);

    paraller_for(
        size_t(0),
        dZ.rows,
        [&](size_t r)
        {
            for (size_t c = 0; c < dZ.cols; c++)
            {
                size_t idx = r * dZ.cols + c;
                dZ.data[idx] = dA.data[idx] * activate_derivative(last_output.data[idx]);
            }
        }
    );

    grad_W = dZ.matmul(last_input, true, false).scaled(1.0f / static_cast<float>(batch_size));
    grad_b = dZ.mean_axis0();

    t++;

    float b1 = beta1;
    float b2 = beta2;
    float lr = learning_rate;
    float b1_t = std::pow(b1, t);
    float b2_t = pow(b2, t);
    float epsilon = 1e-7f;

    paraller_for(
        size_t(0),
        W.data.size(),
        [&](size_t i)
        {
            m_W.data[i] = b1 * m_W.data[i] + (1.0f - b1) * grad_W.data[i];
            v_W.data[i] = b2 * v_W.data[i] + (1.0f - b2) * grad_W.data[i] * grad_W.data[i];

            float m_hat = m_W.data[i] / (1.0f - b1_t);
            float v_hat = v_W.data[i] / (1.0f - b2_t);
            W.data[i] -= lr * (m_hat / (std::sqrt(v_hat) + epsilon));
        }
    );

    for (size_t i = 0; i < b.data.size(); i++)
    {
        m_b.data[i] = b1 * m_b.data[i] + (1.0f - b1) * grad_b.data[i];
        v_b.data[i] = b2 * v_b.data[i] + (1.0f - b2) * grad_b.data[i] * grad_b.data[i];

        float m_hat = m_b.data[i] / (1.0f - b1_t);
        float v_hat = v_b.data[i] / (1.0f - b2_t);
        b.data[i] -= lr * (m_hat / (std::sqrt(v_hat) + epsilon));
    }

    return dZ.matmul(W);
}

Matrix Layer::backward_from_dZ(const Matrix& dZ, float learning_rate, float beta1, float beta2, size_t batch_size)
{
    grad_W = dZ.matmul(last_input, true, false).scaled(1.0f / static_cast<float>(batch_size));
    grad_b = dZ.mean_axis0();

    t++;

    float b1_t = std::pow(beta1, t);
    float b2_t = std::pow(beta2, t);
    float epsilon = 1e-7f;

    paraller_for(
        size_t(0),
        W.data.size(),
        [&](size_t i)
        {
            m_W.data[i] = beta1 * m_W.data[i] + (1.0f - beta1) * grad_W.data[i];
            v_W.data[i] = beta2 * v_W.data[i] + (1.0f - beta2) * grad_W.data[i] * grad_W.data[i];

            float m_hat = m_W.data[i] / (1.0f - b1_t);
            float v_hat = v_W.data[i] / (1.0f - b2_t);
            W.data[i] -= learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon));
        }
    );

    for (size_t i = 0; i < b.data.size(); i++)
    {
        m_b.data[i] = beta1 * m_b.data[i] + (1.0f - beta1) * grad_b.data[i];
        v_b.data[i] = beta2 * v_b.data[i] + (1.0f - beta2) * grad_b.data[i] * grad_b.data[i];

        float m_hat = m_b.data[i] / (1.0f - b1_t);
        float v_hat = v_b.data[i] / (1.0f - b2_t);
        b.data[i] -= learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon));
    }

    Matrix dA_prev = dZ.matmul(W);

    return dA_prev;
}
