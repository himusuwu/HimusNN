#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "Config.hpp"
#include "Metric.hpp"
#include "Network.hpp"
#include "Progress.hpp"
#include "Timing.hpp"
#include <vector>

class Trainer
{
	public:
	  Trainer(const Config& config) : timing(config.window_size), progress(config.bar_width, config.use_color) {}

	  void run(Network& net, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, const Config& config);

	private:
	  Metric metric;
	  Timing timing;
	  Progress progress;

	  double last_mse = 0.0;
	  double last_acc = 0.0;

	  std::chrono::duration<double> last_eta_s{};
      std::chrono::duration<double, std::milli> last_epoch_ms{};
};

#endif