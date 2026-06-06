#ifndef NEURON_HPP
#define NEURON_HPP

#include "Config.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

class Neuron
{
  public:
    Neuron(int inputs_size, ActivationType activation, float leaky_alph, float elu_alpha, float huber_delta);
    float calculate(std::vector<float> inputs);
    float calculate_out_delta(float target, float output, LossType loss);
    float calculate_hidden_delta(float sum);
    const std::vector<float>& get_weights();
    float get_last_output();
    std::vector<float> get_last_inputs();
    float get_delta();
    void update_weights(float delta, float learning_rate);
    void update_bias(float delta, float learning_rate);
    void accumulate_gradients(float delta);
    void apply_batch_update(size_t batch_size, float learning_rate, float momentum);

  private:
    std::vector<float> weights;
    float bias;
    float last_output;
    std::vector<float> last_inputs;
    float delta;

    ActivationType activation;

    float leaky_alpha;
    float elu_alpha;
    float huber_delta;

    std::vector<float> velocity_w;
    float velocity_b;

    std::vector<float> grad_w;
    float grad_b;

    float activate(float x)
    {
        switch (activation)
        {
            case ActivationType::Sigmoid:
                return (1 / (1 + exp(-x)));
            case ActivationType::Tanh:
                return (tanh(x));
            case ActivationType::Relu:
                return (std::max(0.0, static_cast<double>(x)));
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

    float activateDerivative(float y)
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
