#include "DataLoader.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

DataSet DataLoader::loadCSV(const std::string& path, int input_cols, int target_cols, char sep)
{
    DataSet data;
    data.input_size = input_cols;
    data.target_size = target_cols;
    data.num_samples = 0;

    std::ifstream file(path);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + path);
    }

    std::string line;
    int expected_cols = input_cols + target_cols;

    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::stringstream ss(line);
        std::string cell;

        std::vector<float> temp_inputs;
        std::vector<float> temp_targets;
        temp_inputs.reserve(input_cols);
        temp_targets.reserve(target_cols);

        int current_col = 0;

        while (std::getline(ss, cell, sep))
        {
            try
            {
                float value = std::stof(cell);

                if (current_col < input_cols)
                {
                    temp_inputs.push_back(value);
                }
                else if (current_col < expected_cols)
                {
                    temp_targets.push_back(value);
                }
            }
            catch (const std::exception&)
            {
                break;
            }
            current_col++;
        }

        if (temp_inputs.size() == static_cast<size_t>(input_cols)
            && temp_targets.size() == static_cast<size_t>(target_cols))
        {
            data.inputs.insert(data.inputs.end(), temp_inputs.begin(), temp_inputs.end());
            data.targets.insert(data.targets.end(), temp_targets.begin(), temp_targets.end());
            data.num_samples++;
        }
    }

    return data;
}

/*
void DataLoader::normalize(DataSet& data)
{
    if (data.targets.size() != data.inputs.size())
    {
        throw std::length_error("Inputs array size is different than targets array size.");
    }

    std::vector<float> min_vals;
    std::vector<float> max_vals;

    if (data.inputs.empty())
    {
        throw std::length_error("Inputs array is empty.");
    }

    size_t cols = data.inputs[0].size();

    min_vals.resize(cols);
    max_vals.resize(cols);

    float eps = 1e-9;

    for (size_t col = 0; col < cols; col++)
    {
        min_vals[col] = data.inputs[0][col];
        max_vals[col] = data.inputs[0][col];

        for (size_t row = 1; row < data.inputs.size(); row++)
        {
            if (data.inputs[row].size() != cols)
            {
                throw std::length_error("Inputs array wrong size.");
            }

            min_vals[col] = std::min(min_vals[col], data.inputs[row][col]);
            max_vals[col] = std::max(max_vals[col], data.inputs[row][col]);
        }
    }

    for (size_t i = 0; i < data.inputs.size(); i++)
    {
        for (size_t j = 0; j < data.inputs[i].size(); j++)
        {
            data.inputs[i][j] = ((data.inputs[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j] + eps));
        }
    }
}
*/

SplitData DataLoader::split(const DataSet& data, float val_ratio, unsigned seed)
{
    SplitData result;

    result.train.input_size = data.input_size;
    result.train.target_size = data.target_size;
    result.val.input_size = data.input_size;
    result.val.target_size = data.target_size;

    size_t val_count = static_cast<size_t>(val_ratio * data.num_samples);
    size_t train_count = data.num_samples - val_count;

    result.train.num_samples = train_count;
    result.val.num_samples = val_count;

    result.train.inputs.resize(train_count * data.input_size);
    result.train.targets.resize(train_count * data.target_size);

    result.val.inputs.resize(val_count * data.input_size);
    result.val.targets.resize(val_count * data.target_size);

    std::vector<size_t> indicies(data.num_samples);

    for (size_t i = 0; i < data.num_samples; i++)
    {
        indicies[i] = i;
    }

    std::mt19937 rng(seed);
    std::shuffle(indicies.begin(), indicies.end(), rng);

    size_t current_val_idx = 0;
    size_t current_train_idx = 0;

    for (size_t i = 0; i < data.num_samples; i++)
    {
        size_t original_idx = indicies[i];

        size_t src_in_offset = original_idx * data.input_size;
        size_t src_tg_offset = original_idx * data.target_size;

        if (i < val_count)
        {
            size_t dst_in_offset = current_val_idx * data.input_size;
            size_t dst_tg_offset = current_val_idx * data.target_size;

            std::copy(
                data.inputs.begin() + src_in_offset,
                data.inputs.begin() + src_in_offset + data.input_size,
                result.val.inputs.begin() + dst_in_offset
            );
            std::copy(
                data.targets.begin() + src_tg_offset,
                data.targets.begin() + src_tg_offset + data.target_size,
                result.val.targets.begin() + dst_tg_offset
            );

            current_val_idx++;
        }
        else
        {
            size_t dst_in_offset = current_train_idx * data.input_size;
            size_t dst_tg_offset = current_train_idx * data.target_size;

            std::copy(
                data.inputs.begin() + src_in_offset,
                data.inputs.begin() + src_in_offset + data.input_size,
                result.train.inputs.begin() + dst_in_offset
            );
            std::copy(
                data.targets.begin() + src_tg_offset,
                data.targets.begin() + src_tg_offset + data.target_size,
                result.train.targets.begin() + dst_tg_offset
            );

            current_train_idx++;
        }
    }

    return result;
}

DataLoader::NormalizationStats DataLoader::calculate_stats(const DataSet& data)
{
    if (data.num_samples == 0) [[unlikely]]
    {
        throw std::length_error("Dataset is empty.");
    }

    NormalizationStats stats;
    stats.min_vals.resize(data.input_size, 0.0f);
    stats.max_vals.resize(data.input_size, 0.0f);

    for (size_t col = 0; col < data.input_size; col++)
    {
        stats.min_vals[col] = data.inputs[col];
        stats.max_vals[col] = data.inputs[col];

        for (size_t row = 1; row < data.num_samples; row++)
        {
            float val = data.inputs[row * data.input_size + col];
            stats.min_vals[col] = std::min(stats.min_vals[col], val);
            stats.max_vals[col] = std::max(stats.max_vals[col], val);
        }
    }

    return stats;
}

void DataLoader::applyNormalization(DataSet& data, const NormalizationStats& stats)
{
    if (data.num_samples == 0)
    {
        return;
    }

    for (size_t row = 0; row < data.num_samples; row++)
    {
        for (size_t col = 0; col < data.input_size; col++)
        {
            size_t idx = row * data.input_size + col;
            data.inputs[idx] =
                (data.inputs[idx] - stats.min_vals[col]) / (stats.max_vals[col] - stats.min_vals[col] + stats.eps);
        }
    }
}
