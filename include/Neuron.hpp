#ifndef NEURON_HPP
#define NEURON_HPP

#include "Config.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

class Neuron
{
  public:
    Neuron(int inputs_size, ActivationType activation, double leaky_alph, double elu_alpha, double huber_delta);
    double calculate(std::vector<double> inputs);
    double calculate_out_delta(double target, double output, LossType loss);
    double calculate_hidden_delta(double sum);
    const std::vector<double>& get_weights();
    double get_last_output();
    std::vector<double> get_last_inputs();
    double get_delta();
    void update_weights(double delta, double learning_rate);
    void update_bias(double delta, double learning_rate);

  private:
    std::vector<double> weights;
    double bias;
    double last_output;
    std::vector<double> last_inputs;
    double delta;

    ActivationType activation;

    double leaky_alpha;
    double elu_alpha;
    double huber_delta;

    std::vector<double> velocity_w;
    double velocity_b;

    double activate(double x)
    {
        switch (activation)
        {
            case ActivationType::Sigmoid:
                return (1 / (1 + exp(-x)));
            case ActivationType::Tanh:
                return (tanh(x));
            case ActivationType::Relu:
                return (std::max(0.0, x));
            case ActivationType::LeakyRelu:
                return ((x > 0) ? x : leaky_alpha * x);
            case ActivationType::Elu:
                return ((x > 0) ? x : elu_alpha * (exp(x) - 1));
            case ActivationType::Softplus: // TODO SoftPlus
                return 1;
            case ActivationType::Linear:
                return x;
        }
    }

    double activateDerivative(double y)
    {
        switch (activation)
        {
            case ActivationType::Sigmoid:
                return (y * (1 - y));
            case ActivationType::Tanh:
                return (1 - y * y);
            case ActivationType::Relu:
                return ((y > 0) ? 1 : 0);
            case ActivationType::LeakyRelu:
                return ((y > 0) ? 1 : leaky_alpha);
            case ActivationType::Elu:
                return ((y > 0) ? 1 : y + elu_alpha);
            case ActivationType::Softplus:
                return log(1 + exp(y));
            case ActivationType::Linear:
                return 1;
        }
    }
};

#endif
