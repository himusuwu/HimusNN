#include "Trainer.hpp"

#include "Config.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>

void Trainer::run(Network& net, const DataSet& train_data, const DataSet& val_data, const Config& config)
{
    if (train_data.num_samples == 0) [[unlikely]]
    {
        throw std::invalid_argument("Training dataset is empty.");
    }

    for (int epoch = 0; epoch < config.epochs; ++epoch)
    {
        auto start_epoch = std::chrono::steady_clock::now();

        size_t batch_size = static_cast<size_t>(config.batch_size);

        if (batch_size == 0)
        {
            throw std::logic_error("Batch size is 0.");
        }

        for (size_t batch_start = 0; batch_start < train_data.num_samples; batch_start += batch_size)
        {
            size_t current_batch_size = std::min(batch_size, train_data.num_samples - batch_start);
            size_t in_start_idx = batch_start * train_data.input_size;
            size_t in_count = current_batch_size * train_data.input_size;

            size_t tg_start_idx = batch_start * train_data.target_size;
            size_t tg_count = current_batch_size * train_data.target_size;

            std::span<const float> in_span(train_data.inputs.data() + in_start_idx, in_count);
            std::span<const float> tg_span(train_data.targets.data() + tg_start_idx, tg_count);

            net.trainBatch(
                in_span,
                tg_span,
                current_batch_size,
                train_data.input_size,
                train_data.target_size,
                config.beta1,
                config.beta2
            );
        }

        auto stop_epoch = std::chrono::steady_clock::now();

        std::chrono::steady_clock::duration epoch_time = stop_epoch - start_epoch;

        if (epoch % config.status_interval == 0)
        {
            Timing::TimingResult t = timing.update(epoch_time, config.epochs - epoch - 1);
            last_eta_s = t.eta;
            last_epoch_ms = t.epoch_ms;
        }

        if (epoch % config.metric_interval == 0)
        {
            last_mse = metric.mse(net, train_data);
            last_acc = metric.accuracy(net, train_data);

            val_mse = metric.mse(net, val_data);
            val_acc = metric.accuracy(net, val_data);
        }

        progress.update(epoch, config.epochs, last_mse, last_acc, val_mse, val_acc, last_eta_s, last_epoch_ms);
    }

    progress.finish();
}
