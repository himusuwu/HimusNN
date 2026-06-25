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
    int epochs = 100;
    int metric_interval = std::max(1, epochs / 50);

    Config config{
        epochs,
        0,
        1, // metric interval
        100,
        40,
        20,
        true,
        0.001,
        {256, 128, 26},
        ActivationType::LeakyRelu,
        ActivationType::Softmax,
        LossType::CCE,
        0.01f,
        1.0f,
        1.0f,
        512,
        0.9f,
        0.999f
    };

    Network net(
        784,
        config.layers,
        config.learning_rate,
        config.hidden_activation,
        config.output_activation,
        config.loss,
        config.leaky_alpha,
        config.elu_alpha,
        config.huber_delta
    );

    std::string path = "../data/alphabet.csv";
    int input_cols = 784;
    int target_cols = 26;

    std::cout << "[1/4] Loading CSV..." << std::endl;

    DataSet data = DataLoader::loadCSV(path, input_cols, target_cols);

    std::cout << "[2/4] Mixing and spliting data..." << std::endl;

    float val_ratio = 0.2f;
    unsigned seed = 121;

    SplitData split = DataLoader::split(data, val_ratio, seed);

    std::cout << "[3/4] Calculating normalization metrics..." << std::endl;

    DataLoader::NormalizationStats stats = DataLoader::calculate_stats(split.train);

    std::cout << "[4/4] Applying normalization..." << std::endl;

    DataLoader::applyNormalization(split.train, stats);
    DataLoader::applyNormalization(split.val, stats);

    Trainer trainer(config);
    trainer.run(net, split.train, split.val, config);

    std::cout << std::fixed << std::setprecision(6);

    // test
    size_t samples_to_test = std::min<size_t>(10, split.val.num_samples);

    std::cout << std::fixed << std::setprecision(2);

    std::span<const float> test_inputs(split.val.inputs.data(), samples_to_test * split.val.input_size);
    Matrix test_matrix(samples_to_test, split.val.input_size, test_inputs);

    Matrix out_matrix = net.predictBatch(test_matrix);

    for (size_t i = 0; i < samples_to_test; ++i)
    {
        int predicted_label = 0;
        int actual_label = 0;
        float confidence = 0.0f;

        if (split.val.target_size == 1)
        {
            // Binary
            predicted_label = (out_matrix(i, 0) > 0.5f) ? 1 : 0;
            actual_label = (split.val.targets[i * split.val.target_size] > 0.5f) ? 1 : 0;
            confidence = (predicted_label == 1) ? out_matrix(i, 0) : (1.0f - out_matrix(i, 0));
        }
        else
        {
            // Multiclass
            float max_pred = -1e9f;
            float max_target = -1e9f;

            for (size_t j = 0; j < split.val.target_size; j++)
            {
                if (out_matrix(i, j) > max_pred)
                {
                    max_pred = out_matrix(i, j);
                    predicted_label = static_cast<int>(j);
                    confidence = out_matrix(i, j);
                }

                if (split.val.targets[i * split.val.target_size + j] > max_target)
                {
                    max_target = split.val.targets[i * split.val.target_size + j];
                    actual_label = static_cast<int>(j);
                }
            }
        }

        if (split.val.target_size > 10)
        {
            std::cout << "Predicted: " << static_cast<char>('A' + predicted_label) << "'";
        }
        else
        {
            std::cout << "Predicted: " << predicted_label;
        }

        std::cout << " | Confidence: " << (confidence * 100.0f) << "%";

        if (predicted_label == actual_label)
        {
            std::cout << " OK\n";
        }
        else
        {
            std::cout << " ERR (" << actual_label << ")\n";
        }
    }

    return 0;
}
