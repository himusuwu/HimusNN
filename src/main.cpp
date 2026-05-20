#include <iostream>
#include <vector>
#include "../include/Network.hpp"

int main()
{
    // Parity 3-bit: output = 1, gdy liczba jedynek jest nieparzysta
    Network net(3, {6, 1}); // wejscie 3 -> ukryta 6 -> wyjscie 1

    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 1.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 0.0},
        {1.0, 1.0, 1.0}
    };

    std::vector<std::vector<double>> targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0},
        {1.0},
        {0.0},
        {0.0},
        {1.0}
    };

    int epochs = 50000;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            net.train(inputs[i], targets[i]);
        }
    }

    // test
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        std::vector<double> out = net.predict(inputs[i]);
        double y = out[0];
        int pred = (y > 0.5) ? 1 : 0;

        std::cout << inputs[i][0] << inputs[i][1] << inputs[i][2]
                  << " -> " << pred << " (raw=" << y << ")\n";
    }

    return 0;
}