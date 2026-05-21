#include "Metric.hpp"

double Metric::mse(Network& net, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets)
{
	sum_squared_error = 0.0;

    for(size_t i = 0; i < inputs.size(); i++)
    {
        std::vector<double> out = net.predict(inputs[i]);
        double diff = targets[i][0] - out[0];

        sum_squared_error += diff * diff;
    }

    return sum_squared_error / inputs.size();
}

double Metric::accuracy(Network& net, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets)
{
	correct = 0;

    for(size_t i = 0; i < inputs.size(); i++)
    {
        std::vector<double> out = net.predict(inputs[i]);
        double y = out[0];

        int pred = (y > 0.5) ? 1 : 0;

        if(pred == static_cast<int>(targets[i][0]))
        {
            correct++;
        }
    }

    return static_cast<double>(correct) / inputs.size();
}