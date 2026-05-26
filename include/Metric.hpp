#ifndef METRIC_HPP
#define METRIC_HPP

#include "Network.hpp"

class Metric
{
  public:
    double
    mse(Network& net, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets);

    double accuracy(
        Network& net, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets
    );

  private:
    double sum_squared_error = 0.0;
    size_t correct = 0;
};

#endif
