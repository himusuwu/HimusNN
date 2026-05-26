#include "Config.hpp"
#include "DataLoader.hpp"
#include "DataSet.hpp"
#include "Network.hpp"
#include "Trainer.hpp"

#include <algorithm>
#include <iomanip>
#include <ios>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

int main()
{
    int epochs = 50000;
    int metric_interval = std::max(1, epochs / 100);

    Config config{
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
    Network net(
        3,
        config.layers,
        config.learning_rate,
        config.activation,
        config.loss,
        config.leaky_alpha,
        config.elu_alpha,
        config.huber_delta
    ); // wejscie 3 -> ukryta 6 -> wyjscie 1

    std::string path = "../data/parity.csv";
    int input_cols = 3;
    int target_cols = 1;

    DataSet data = DataLoader::loadCSV(path, input_cols, target_cols);

    double val_ratio = 0.2;
    unsigned seed = 121;

    SplitData split = DataLoader::split(data, val_ratio, seed);

    DataLoader::NormalizationStats stats = DataLoader::calculate_stats(split.train);
    DataLoader::applyNormalization(split.train, stats);
    DataLoader::applyNormalization(split.val, stats);

    Trainer trainer(config);
    trainer.run(net, split.train.inputs, split.train.targets, split.val.inputs, split.val.targets, config);

    std::cout << std::fixed << std::setprecision(6);

    // test
    for (size_t i = 0; i < split.train.inputs.size(); ++i)
    {
        std::vector<double> out = net.predict(split.train.inputs[i]);
        double y = out[0];
        int pred = (y > 0.5) ? 1 : 0;

        std::cout << std::defaultfloat << split.train.inputs[i][0] << split.train.inputs[i][1]
                  << split.train.inputs[i][2] << " -> " << pred << " (raw=" << std::fixed << y << ")\n";
    }

    return 0;
}
