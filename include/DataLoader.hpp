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
        std::vector<double> min_vals;
        std::vector<double> max_vals;

        double eps = 1e-9;
    };

    static DataSet loadCSV(const std::string& path, int input_cols, int target_cols, char sep = ',');
    static void normalize(DataSet& data);
    static SplitData split(const DataSet& data, double val_ratio, unsigned seed);
    static NormalizationStats calculate_stats(const DataSet& data);
    static void applyNormalization(DataSet& data, const NormalizationStats& stats);
};

#endif
