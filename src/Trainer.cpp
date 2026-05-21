#include "Trainer.hpp"
#include "Config.hpp"
#include <chrono>

void Trainer::run(Network& net, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, const Config& config)
{
	for (int epoch = 0; epoch < config.epochs; ++epoch)
    {
        auto start_epoch = std::chrono::steady_clock::now();

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            net.train(inputs[i], targets[i]);
        }

        auto stop_epoch = std::chrono::steady_clock::now();

		std::chrono::steady_clock::duration epoch_time = stop_epoch - start_epoch;

		if(epoch % config.status_interval == 0)
        {
            Timing::TimingResult t = timing.update(epoch_time, config.epochs - epoch - 1);
			last_eta_s = t.eta;
			last_epoch_ms = t.epoch_ms;
        }

		if(epoch % config.metric_interval == 0)
		{
			last_mse = metric.mse(net, inputs, targets);
			last_acc = metric.accuracy(net, inputs, targets);
		}

		progress.update(epoch, config.epochs, last_mse,last_acc, last_eta_s, last_epoch_ms);
    }

	progress.finish();
}