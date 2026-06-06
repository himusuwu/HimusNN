#include "../include/Network.hpp"

#include "Config.hpp"
#include "Matrix.hpp"

#include <cstddef>
#include <vector>

static Matrix make_batch_matrix(const std::vector<std::vector<float>>& v, size_t start, size_t end)
{
    size_t rows = end - start;
    size_t cols = v[start].size();

    Matrix out(rows, cols);

    for (size_t r = 0; r < rows; r++)
    {
        for (size_t c = 0; c < cols; c++)
        {
            out(r, c) = v[start + r][c];
        }
    }

    return out;
}

Network::Network(
    int input_size,
    std::vector<int> layers_sizes,
    float learning_rate,
    ActivationType hidden_activation,
    ActivationType output_activation,
    LossType loss,
    float leaky_alpha,
    float elu_alpha,
    float huber_delta
) :
    learning_rate(learning_rate), hidden_activation(hidden_activation), output_activation(output_activation), loss(loss)
{
    int prev_size = input_size;

    for (size_t i = 0; i < layers_sizes.size(); i++)
    {
        ActivationType current_act = (i == layers_sizes.size() - 1) ? output_activation : hidden_activation;

        layers.emplace_back(layers_sizes[i], prev_size, current_act, leaky_alpha, elu_alpha, huber_delta);
        prev_size = layers_sizes[i];
    }
}

std::vector<float> Network::predict(const std::vector<float>& inputs)
{
    Matrix X = Matrix::from_vector(inputs, true);

    for (Layer& layer : layers)
    {
        X = layer.forward(X);
    }

    return X.row(0);
}

void Network::train(const std::vector<float>& inputs, const std::vector<float>& targets)
{
    std::vector<std::vector<float>> in{inputs};
    std::vector<std::vector<float>> out{targets};

    trainBatch(in, out, 0, 1, 0.0f, 0.0f);
}

void Network::trainBatch(
    const std::vector<std::vector<float>>& inputs,
    const std::vector<std::vector<float>>& targets,
    size_t batch_start,
    size_t batch_end,
    float beta1,
    float beta2
)
{
    size_t batch_size = batch_end - batch_start;

    Matrix X = make_batch_matrix(inputs, batch_start, batch_end);
    Matrix Y = make_batch_matrix(targets, batch_start, batch_end);

    Matrix A = X;

    for (Layer& layer : layers)
    {
        A = layer.forward(A);
    }

    // BCE + Sigmoid: dZ = (output - target)
    Matrix dZ = A.sub(Y);

    Matrix dA = layers.back().backward_from_dZ(dZ, learning_rate, beta1, beta2, batch_size);

    for (int i = static_cast<int>(layers.size()) - 2; i >= 0; i--)
    {
        dA = layers[i].backward(dA, learning_rate, beta1, beta2, batch_size);
    }
}
