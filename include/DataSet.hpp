#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>

struct DataSet
{
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;
};

struct SplitData
{
    DataSet train;
    DataSet val;
};

#endif
