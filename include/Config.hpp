#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <vector>

enum class ActivationType
{
    Sigmoid,   // σ(x) = 1 / (1 + exp(-x))
    Tanh,      // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Relu,      // ReLu(x) = max(0, x)
    LeakyRelu, // LReLu(x) = (x > 0) ? x : ax
    Elu,       // Elu(x) = (x > 0) ? x : a(exp(x) - 1)
    Softplus,  // SPlus(x) = log(1 + exp(x))
    Linear     // f(x) = x
};

enum class LossType
{
    MSE,         // Mean Squared Error: L = (t - y)²
    BCE,         // Binary Cross-Entropy: L = −(tlog(y) + (1 − t)log(1 − y))
    MAE,         // Mean Absolute Error: L = |t - y|
    Huber,       // MSE for small errors, MAE for big
    CrossEntropy // Softmax + CE
};

struct Config
{
    int epochs;
    int log_interval;
    int metric_interval;
    int status_interval;

    size_t bar_width;
    size_t window_size;

    bool use_color;

    double learning_rate;

    std::vector<int> layers;

    ActivationType activation;
    LossType loss;

    double leaky_alpha;
    double elu_alpha;
    double huber_delta;

    size_t batch_size;
    double momentum;
    double beta1;
    double beta2;
};

#endif
