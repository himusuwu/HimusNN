#include "Metric.hpp"

#include "Matrix.hpp"

#include <span>
#include <vector>

static Matrix make_eval_matrix(const std::vector<std::vector<float>>& v)
{
    if (v.empty()) [[unlikely]]
    {
        return Matrix(0, 0);
    }

    size_t rows = v.size();
    size_t cols = v[0].size();

    Matrix out(rows, cols);

    for (size_t r = 0; r < rows; r++)
    {
        for (size_t c = 0; c < cols; c++)
        {
            out(r, c) = v[r][c];
        }
    }

    return out;
}

float Metric::mse(Network& net, const DataSet& data)
{
    if (data.num_samples == 0) [[unlikely]]
    {
        return 0.0f;
    }

    sum_squared_error = 0.0f;

    std::span<const float> inputs_span(data.inputs.data(), data.inputs.size());

    Matrix X_batch(data.num_samples, data.input_size, inputs_span);
    Matrix out_batch = net.predictBatch(X_batch);

    for (size_t i = 0; i < data.num_samples; i++)
    {
        for (size_t j = 0; j < data.target_size; j++)
        {
            float target_val = data.targets[i * data.target_size + j];
            float diff = target_val - out_batch(i, j);
            sum_squared_error += diff * diff;
        }
    }

    return sum_squared_error / data.num_samples;
}

float Metric::accuracy(Network& net, const DataSet& data)
{
    if (data.num_samples == 0) [[unlikely]]
    {
        return 0.0f;
    }

    correct = 0;

    std::span<const float> inputs_span(data.inputs.data(), data.inputs.size());

    Matrix X_batch(data.num_samples, data.input_size, inputs_span);
    Matrix out_batch = net.predictBatch(X_batch);

    for (size_t i = 0; i < data.num_samples; i++)
    {
        if (data.target_size == 1)
        {
            float y = out_batch(i, 0);
            int pred = (y > 0.5) ? 1 : 0;

            if (pred == static_cast<int>(data.targets[i * data.target_size + 0]))
            {
                correct++;
            }
        }
        else
        {
            float max_pred_val = -1e9f;
            int best_pred_idx = 0;

            float max_target_val = -1e9f;
            int best_target_idx = 0;

            for (size_t j = 0; j < data.target_size; j++)
            {
                if (out_batch(i, j) > max_pred_val)
                {
                    max_pred_val = out_batch(i, j);
                    best_pred_idx = static_cast<int>(j);
                }

                if (data.targets[i * data.target_size + j] > max_target_val)
                {
                    max_target_val = data.targets[i * data.target_size + j];
                    best_target_idx = static_cast<int>(j);
                }
            }

            if (best_pred_idx == best_target_idx)
            {
                correct++;
            }
        }
    }

    return static_cast<float>(correct) / data.num_samples;
}
