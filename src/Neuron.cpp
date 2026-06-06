#include "../include/Neuron.hpp"

#include "Config.hpp"

#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

Neuron::Neuron(int input_size, ActivationType activation, float leaky_alpha, float elu_alpha, float huber_delta) :
    activation(activation), leaky_alpha(leaky_alpha), elu_alpha(elu_alpha), huber_delta(huber_delta)
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (int i = 0; i < input_size; i++)
    {
        weights.push_back(dist(rng));
    }

    bias = dist(rng);
    last_output = 0.0f;

    grad_w.resize(weights.size());
    velocity_w.resize(weights.size());
    grad_b = 0.0f;
    velocity_b = 0.0f;
}

// float sum = (w1 * x1) + (w2 * x2) + bias;

float Neuron::calculate(std::vector<float> inputs)
{
    last_inputs.clear();

    for (float& input : inputs)
    {
        last_inputs.push_back(input);
    }

    float sum = 0.0f;

    if (inputs.size() == weights.size())
    {
        for (size_t i = 0; i < weights.size(); i++)
        {
            sum += weights[i] * inputs[i];
        }

        sum += bias;
    }

    last_output = activate(sum);

    return activate(sum);
}

float Neuron::calculate_out_delta(float target, float output, LossType loss)
{
    switch (loss)
    {
        case LossType::MSE:
        {
            delta = (target - output) * activateDerivative(output);
            break;
        }
        case LossType::BCE:
        {
            if (activation == ActivationType::Sigmoid)
            {
                delta = (target - output);
            }
            else
            {
                throw std::invalid_argument(
                    "Model configuration error: Binary Cross Entropy (BCE) loss requires "
                    "a Sigmoid activation function on the final layer to ensure outputs "
                    "fall within the [0, 1] range."
                );
            }
            break;
        }
        case LossType::MAE:
        {
            float e = output - target;
            float grad = (e > 0) ? 1.0f : (e < 0) ? -1.0f : 0.0f;

            delta = -grad * activateDerivative(output);
            break;
        }
        case LossType::Huber:
        {
            float e = output - target;
            float abs_e = std::abs(e);
            float grad = (abs_e <= huber_delta) ? e : huber_delta * ((e > 0) ? 1.0f : -1.0f);

            delta = -grad * activateDerivative(output);
            break;
        }
        case LossType::CrossEntropy: // TODO CrossEntropy
            break;
    }

    return delta;
}

float Neuron::calculate_hidden_delta(float sum)
{
    delta = sum * activateDerivative(last_output);

    return delta;
}

const std::vector<float>& Neuron::get_weights()
{
    return weights;
}

float Neuron::get_last_output()
{
    return last_output;
}

std::vector<float> Neuron::get_last_inputs()
{
    return last_inputs;
}

float Neuron::get_delta()
{
    return delta;
}

void Neuron::update_weights(float delta, float learning_rate)
{
    for (int i = 0; i < weights.size(); i++)
    {
        weights[i] += learning_rate * delta * last_inputs[i];
    }
}

void Neuron::update_bias(float delta, float learning_rate)
{
    bias += learning_rate * delta;
}

void Neuron::accumulate_gradients(float delta)
{
    for (int i = 0; i < weights.size(); i++)
    {
        grad_w[i] += delta * last_inputs[i];
    }

    grad_b += delta;
}

void Neuron::apply_batch_update(size_t batch_size, float learning_rate, float momentum)
{
    for (int i = 0; i < weights.size(); i++)
    {
        velocity_w[i] = momentum * velocity_w[i] + (grad_w[i] / batch_size);
        weights[i] += learning_rate * velocity_w[i];
    }

    velocity_b = momentum * velocity_b + (grad_b / batch_size);
    bias += learning_rate * velocity_b;

    std::fill(grad_w.begin(), grad_w.end(), 0.0f);
    grad_b = 0.0f;
}
