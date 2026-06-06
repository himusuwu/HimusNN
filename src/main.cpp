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
    int epochs = 5000;
    int metric_interval = std::max(1, epochs / 50);

    Config config{
        epochs,
        0,
        metric_interval,
        100,
        40,
        20,
        true,
        0.005,
        {32, 16, 1},
        ActivationType::Relu,
        ActivationType::Sigmoid,
        LossType::BCE,
        0.01f,
        1.0f,
        1.0f,
        1024,
        0.9f,
        0.999f
    };

    // Parity 12-bit: output = 1 gdy liczba jedynek jest nieparzysta
    Network net(
        12,
        config.layers,
        config.learning_rate,
        config.hidden_activation,
        config.output_activation,
        config.loss,
        config.leaky_alpha,
        config.elu_alpha,
        config.huber_delta
    ); // wejscie 3 -> ukryta 6 -> wyjscie 1

    std::string path = "../data/parity.csv";
    int input_cols = 12;
    int target_cols = 1;

    DataSet data = DataLoader::loadCSV(path, input_cols, target_cols);

    float val_ratio = 0.2f;
    unsigned seed = 121;

    SplitData split = DataLoader::split(data, val_ratio, seed);

    DataLoader::NormalizationStats stats = DataLoader::calculate_stats(split.train);
    DataLoader::applyNormalization(split.train, stats);
    DataLoader::applyNormalization(split.val, stats);

    Trainer trainer(config);
    trainer.run(net, split.train.inputs, split.train.targets, split.val.inputs, split.val.targets, config);

    std::cout << std::fixed << std::setprecision(6);

    // test
    for (size_t i = 0; i < std::min<size_t>(10, split.train.inputs.size()); ++i)
    {
        std::vector<float> out = net.predict(split.train.inputs[i]);
        float y = out[0];
        int pred = (y > 0.5f) ? 1 : 0;

        for (size_t j = 0; j < split.train.inputs[i].size(); ++j)
        {
            std::cout << std::defaultfloat << split.train.inputs[i][j];
        }
        std::cout << " -> " << pred << " (raw=" << std::fixed << y << ")\n";
    }

    return 0;
}
