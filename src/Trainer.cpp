#include "Trainer.hpp"

#include "Config.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>

void Trainer::run(
    Network& net,
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets,
    const std::vector<std::vector<double>>& val_inputs,
    const std::vector<std::vector<double>>& val_targets,
    const Config& config
)
{
    for (int epoch = 0; epoch < config.epochs; ++epoch)
    {
        auto start_epoch = std::chrono::steady_clock::now();

        size_t batch_size = static_cast<size_t>(config.batch_size);
        size_t total = inputs.size();

        if (batch_size == 0)
        {
            throw std::logic_error("Batch size is 0.");
        }

        for (size_t batch_start = 0; batch_start < total; batch_start += batch_size)
        {
            size_t batch_end = std::min(batch_start + batch_size, total);

            net.trainBatch(inputs, targets, batch_start, batch_end, config.momentum);
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
            last_mse = metric.mse(net, inputs, targets);
            last_acc = metric.accuracy(net, inputs, targets);

            val_mse = metric.mse(net, val_inputs, val_targets);
            val_acc = metric.accuracy(net, val_inputs, val_targets);
        }

        progress.update(epoch, config.epochs, last_mse, last_acc, val_mse, val_acc, last_eta_s, last_epoch_ms);
    }

    progress.finish();
}
