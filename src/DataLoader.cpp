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

    std::ifstream file(path);

    int expected_cols = input_cols + target_cols;

    if (file.is_open())
    {
        std::string line;

        while (std::getline(file, line))
        {
            std::vector<float> input;
            std::vector<float> target;

            std::stringstream ss(line);
            std::string cell;

            int current_col = 0;

            while (std::getline(ss, cell, sep))
            {
                float value = std::stod(cell);

                if (current_col < input_cols)
                {
                    input.push_back(value);
                }
                else if (current_col < input_cols + target_cols)
                {
                    target.push_back(value);
                }

                current_col++;
            }

            if (current_col >= expected_cols)
            {
                data.inputs.push_back(input);
                data.targets.push_back(target);
            }
        }
    }

    return data;
}

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

SplitData DataLoader::split(const DataSet& data, float val_ratio, unsigned seed)
{
    SplitData result;

    std::vector<size_t> indicies;

    for (size_t i = 0; i < data.inputs.size(); i++)
    {
        indicies.push_back(i);
    }

    std::mt19937 rng(seed);

    std::shuffle(indicies.begin(), indicies.end(), rng);

    size_t val_count = static_cast<size_t>(val_ratio * data.inputs.size());

    for (size_t i = 0; i < indicies.size(); i++)
    {
        if (i < val_count)
        {
            result.val.inputs.push_back(data.inputs[indicies[i]]);
            result.val.targets.push_back(data.targets[indicies[i]]);
        }
        else
        {
            result.train.inputs.push_back(data.inputs[indicies[i]]);
            result.train.targets.push_back(data.targets[indicies[i]]);
        }
    }

    return result;
}

DataLoader::NormalizationStats DataLoader::calculate_stats(const DataSet& data)
{
    std::vector<float> min_vals;
    std::vector<float> max_vals;

    if (data.inputs.empty())
    {
        throw std::length_error("Inputs array is empty.");
    }

    size_t cols = data.inputs[0].size();

    min_vals.resize(cols);
    max_vals.resize(cols);

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

    return NormalizationStats{min_vals, max_vals};
}

void DataLoader::applyNormalization(DataSet& data, const NormalizationStats& stats)
{
    if (data.inputs.empty())
    {
        return;
    }

    size_t cols = stats.min_vals.size();

    if (stats.max_vals.size() != cols)
    {
        throw std::logic_error("Max vals array error.");
    }

    for (size_t row = 0; row < data.inputs.size(); row++)
    {
        if (data.inputs[row].size() != cols)
        {
            throw std::length_error("Inputs array wrong size");
        }

        for (size_t col = 0; col < cols; col++)
        {
            data.inputs[row][col] =
                (data.inputs[row][col] - stats.min_vals[col]) / (stats.max_vals[col] - stats.min_vals[col] + stats.eps);
        }
    }
}
