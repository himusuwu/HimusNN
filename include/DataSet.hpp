#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>

struct DataSet
{
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;
};

struct SplitData
{
    DataSet train;
    DataSet val;
};

#endif
