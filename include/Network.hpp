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
        float learning_rate,
        ActivationType hidden_activation,
        ActivationType output_activation,
        LossType loss,
        float leaky_alpha,
        float elu_alpha,
        float huber_delta
    );
    [[nodiscard]] std::vector<float> predict(const std::vector<float>& inputs);
    void train(const std::vector<float>& inputs, const std::vector<float>& targets);
    void trainBatch(
        const std::vector<std::vector<float>>& inputs,
        const std::vector<std::vector<float>>& targets,
        size_t batch_start,
        size_t batch_end,
        float beta1,
        float beta2
    );
    void save(const std::string& path) const;
    void load(const std::string& path);

  private:
    std::vector<Layer> layers;
    float learning_rate;

    ActivationType hidden_activation;
    ActivationType output_activation;
    LossType loss;
};

#endif
