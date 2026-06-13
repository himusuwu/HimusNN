#ifndef DATASET_HPP
#define DATASET_HPP

#include <cstddef>
#include <vector>

struct DataSet
{
    std::vector<float> inputs;
    std::vector<float> targets;

    size_t num_samples = 0;
    size_t input_size = 0;
    size_t target_size = 0;
};

struct SplitData
{
    DataSet train;
    DataSet val;
};

#endif
