#include "Metric.hpp"

float Metric::mse(
    Network& net, const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets
)
{
    sum_squared_error = 0.0f;

    for (size_t i = 0; i < inputs.size(); i++)
    {
        std::vector<float> out = net.predict(inputs[i]);
        float diff = targets[i][0] - out[0];

        sum_squared_error += diff * diff;
    }

    return sum_squared_error / inputs.size();
}

float Metric::accuracy(
    Network& net, const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets
)
{
    correct = 0;

    for (size_t i = 0; i < inputs.size(); i++)
    {
        std::vector<float> out = net.predict(inputs[i]);
        float y = out[0];

        int pred = (y > 0.5) ? 1 : 0;

        if (pred == static_cast<int>(targets[i][0]))
        {
            correct++;
        }
    }

    return static_cast<float>(correct) / inputs.size();
}
