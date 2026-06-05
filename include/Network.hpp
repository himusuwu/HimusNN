#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "Config.hpp"
#include "Layer.hpp"

#include <vector>

class Network
{
  public:
    Network(
        int input_size,
        std::vector<int> layers_sizes,
        double learning_rate,
        ActivationType hidden_activation,
        ActivationType output_activation,
        LossType loss,
        double leaky_alpha,
        double elu_alpha,
        double huber_delta
    );
    std::vector<double> predict(const std::vector<double>& inputs);
    void train(const std::vector<double>& inputs, const std::vector<double>& targets);
    void trainBatch(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets,
        size_t batch_start,
        size_t batch_end,
        double momentum
    );
    void save(const std::string& path) const;
    void load(const std::string& path);

  private:
    std::vector<Layer> layers;
    double learning_rate;

    ActivationType hidden_activation;
    ActivationType output_activation;
    LossType loss;
};

#endif
