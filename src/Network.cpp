#include "../include/Network.hpp"

#include <vector>

Network::Network(
    int input_size,
    std::vector<int> layers_sizes,
    double learning_rate,
    ActivationType activation,
    LossType loss,
    double leaky_alpha,
    double elu_alpha,
    double huber_delta
) :
    learning_rate(learning_rate), activation(activation), loss(loss)
{
    int prev_size = input_size;

    for (int& size : layers_sizes)
    {
        layers.push_back(Layer(size, prev_size, activation, leaky_alpha, elu_alpha, huber_delta));
        prev_size = size;
    }
}

std::vector<double> Network::predict(const std::vector<double>& inputs)
{
    std::vector<double> current = inputs;

    for (Layer& layer : layers)
    {
        current = layer.forward(current);
    }

    return current;
}

void Network::train(const std::vector<double>& inputs, const std::vector<double>& targets)
{
    predict(inputs);

    double sum = 0.0;

    std::vector<Neuron>& neurons = layers.back().getNeurons();

    if (targets.size() == layers.back().getNeurons().size())
    {
        for (int i = 0; i < targets.size(); i++)
        {
            double target = targets[i];
            double output = neurons[i].get_last_output();

            neurons[i].calculate_out_delta(target, output, loss);
        }
    }
    else
    {
        return;
    }

    if (layers.size() >= 2)
    {
        int start = static_cast<int>(layers.size()) - 2;

        for (int i = start; i >= 0; i--)
        {
            Layer& current_layer = layers[i];
            std::vector<Neuron>& current_neurons = current_layer.getNeurons();

            Layer& next_layer = layers[i + 1];
            std::vector<Neuron>& next_neurons = next_layer.getNeurons();

            for (size_t j = 0; j < current_neurons.size(); j++)
            {
                sum = 0.0;

                for (size_t k = 0; k < next_neurons.size(); k++)
                {
                    sum += next_neurons[k].get_weights()[j] * next_neurons[k].get_delta();
                }

                current_neurons[j].calculate_hidden_delta(sum);
            }
        }
    }
    else
    {
        return;
    }

    for (Layer& layer : layers)
    {
        for (Neuron& neuron : layer.getNeurons())
        {
            double delta = neuron.get_delta();

            neuron.update_weights(delta, learning_rate);
            neuron.update_bias(delta, learning_rate);
        }
    }
}

void Network::trainBatch(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets,
    size_t batch_start,
    size_t batch_end,
    double momentum
)
{
    for (size_t sample = batch_start; sample < batch_end; sample++)
    {
        predict(inputs[sample]);

        double sum = 0.0;

        std::vector<Neuron>& neurons = layers.back().getNeurons();

        for (int out_idx = 0; out_idx < targets[sample].size(); out_idx++)
        {
            double target = targets[sample][out_idx];
            double output = neurons[out_idx].get_last_output();

            neurons[out_idx].calculate_out_delta(target, output, loss);
        }

        if (layers.size() >= 2)
        {
            int start = static_cast<int>(layers.size()) - 2;

            for (int i = start; i >= 0; i--)
            {
                Layer& current_layer = layers[i];
                std::vector<Neuron>& current_neurons = current_layer.getNeurons();

                Layer& next_layer = layers[i + 1];
                std::vector<Neuron>& next_neurons = next_layer.getNeurons();

                for (size_t j = 0; j < current_neurons.size(); j++)
                {
                    sum = 0.0;

                    for (size_t k = 0; k < next_neurons.size(); k++)
                    {
                        sum += next_neurons[k].get_weights()[j] * next_neurons[k].get_delta();
                    }

                    current_neurons[j].calculate_hidden_delta(sum);
                }
            }
        }
        else
        {
            return;
        }

        for (Layer& layer : layers)
        {
            for (Neuron& neuron : layer.getNeurons())
            {
                double delta = neuron.get_delta();

                neuron.accumulate_gradients(delta);
            }
        }
    }

    for (Layer& layer : layers)
    {
        for (Neuron& neuron : layer.getNeurons())
        {
            double delta = neuron.get_delta();

            neuron.apply_batch_update(batch_end - batch_start, learning_rate, momentum);
        }
    }
}
