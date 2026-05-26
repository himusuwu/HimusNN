#include "../include/Neuron.hpp"

#include "Config.hpp"

#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

Neuron::Neuron(int input_size, ActivationType activation, double leaky_alpha, double elu_alpha, double huber_delta) :
    activation(activation), leaky_alpha(leaky_alpha), elu_alpha(elu_alpha), huber_delta(huber_delta)
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for (int i = 0; i < input_size; i++)
    {
        weights.push_back(dist(rng));
    }

    bias = dist(rng);
    last_output = 0.0;
}

// double sum = (w1 * x1) + (w2 * x2) + bias;

double Neuron::calculate(std::vector<double> inputs)
{
    last_inputs.clear();

    for (double& input : inputs)
    {
        last_inputs.push_back(input);
    }

    double sum = 0.0;

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

double Neuron::calculate_out_delta(double target, double output, LossType loss)
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
            double e = output - target;
            double grad = (e > 0) ? 1.0 : (e < 0) ? -1.0 : 0.0;

            delta = -grad * activateDerivative(output);
            break;
        }
        case LossType::Huber:
        {
            double e = output - target;
            double abs_e = std::abs(e);
            double grad = (abs_e <= huber_delta) ? e : huber_delta * ((e > 0) ? 1.0 : -1.0);

            delta = -grad * activateDerivative(output);
            break;
        }
        case LossType::CrossEntropy: // TODO CrossEntropy
            break;
    }

    return delta;
}

double Neuron::calculate_hidden_delta(double sum)
{
    delta = sum * activateDerivative(last_output);

    return delta;
}

const std::vector<double>& Neuron::get_weights()
{
    return weights;
}

double Neuron::get_last_output()
{
    return last_output;
}

std::vector<double> Neuron::get_last_inputs()
{
    return last_inputs;
}

double Neuron::get_delta()
{
    return delta;
}

void Neuron::update_weights(double delta, double learning_rate)
{
    for (int i = 0; i < weights.size(); i++)
    {
        weights[i] += learning_rate * delta * last_inputs[i];
    }
}

void Neuron::update_bias(double delta, double learning_rate)
{
    bias += learning_rate * delta;
}
