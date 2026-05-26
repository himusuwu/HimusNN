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
        ActivationType activation,
        LossType loss,
        double leaky_alpha,
        double elu_alpha,
        double huber_delta
    );
    std::vector<double> predict(std::vector<double> inputs);
    void train(std::vector<double> inputs, std::vector<double> targets);
    void trainBatch(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs);
    void save(const std::string& path) const;
    void load(const std::string& path);

  private:
    std::vector<Layer> layers;
    double learning_rate;

    ActivationType activation;
    LossType loss;
};

#endif
