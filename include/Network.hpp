#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "Config.hpp"
#include "Layer.hpp"

#include <span>
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
    [[nodiscard]] Matrix predictBatch(const Matrix& batch_inputs);
    void train(const std::vector<float>& inputs, const std::vector<float>& targets);
    void trainBatch(
        std::span<const float> inputs_flat,
        std::span<const float> targets_flat,
        size_t batch_size,
        size_t input_size,
        size_t target_size,
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
