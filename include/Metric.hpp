#ifndef METRIC_HPP
#define METRIC_HPP

#include "Network.hpp"

class Metric
{
  public:
    float
    mse(Network& net, const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets);

    float accuracy(
        Network& net, const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets
    );

  private:
    float sum_squared_error = 0.0f;
    size_t correct = 0;
};

#endif
