#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include "DataSet.hpp"

#include <string>
#include <vector>

class DataLoader
{
  public:
    struct NormalizationStats
    {
        std::vector<float> min_vals;
        std::vector<float> max_vals;

        float eps = 1e-9f;
    };

    static DataSet loadCSV(const std::string& path, int input_cols, int target_cols, char sep = ',');
    static void normalize(DataSet& data);
    static SplitData split(const DataSet& data, float val_ratio, unsigned seed);
    static NormalizationStats calculate_stats(const DataSet& data);
    static void applyNormalization(DataSet& data, const NormalizationStats& stats);
};

#endif
