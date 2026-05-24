#include <ios>
#include <iostream>
#include <ostream>
#include <vector>
#include <iomanip>
#include <algorithm>

#include "Network.hpp"
#include "Config.hpp"
#include "Trainer.hpp"

int main()
{
    int epochs = 50000;
    int metric_interval = std::max(1, epochs / 100);

    Config config
    {
        epochs, 
        0, 
        metric_interval, 
        10,
        40,
        20,
        true,
        0.5,
        {6, 1},
        ActivationType::Sigmoid,
        LossType::BCE,
        0.01,
        1.0,
        1.0
    };

    // Parity 3-bit: output = 1 gdy liczba jedynek jest nieparzysta
    Network net(3, config.layers, config.learning_rate, config.activation, config.loss, config.leaky_alpha, config.elu_alpha, config.huber_delta); // wejscie 3 -> ukryta 6 -> wyjscie 1

    std::vector<std::vector<double>> inputs = 
    {
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 1.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 0.0},
        {1.0, 1.0, 1.0}
    };

    std::vector<std::vector<double>> targets = 
    {
        {0.0},
        {1.0},
        {1.0},
        {0.0},
        {1.0},
        {0.0},
        {0.0},
        {1.0}
    };

    Trainer trainer(config);
    trainer.run(net, inputs, targets, config);

    std::cout << std::fixed << std::setprecision(6);

    // test
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        std::vector<double> out = net.predict(inputs[i]);
        double y = out[0];
        int pred = (y > 0.5) ? 1 : 0;

        std::cout << std::defaultfloat << inputs[i][0] << inputs[i][1] << inputs[i][2]
                  << " -> " << pred << " (raw=" << std::fixed << y << ")\n";
    }

    return 0;
}