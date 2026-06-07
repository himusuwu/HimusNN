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
    explicit Trainer(const Config& config) : timing(config.window_size), progress(config.bar_width, config.use_color) {}

    void
    run(Network& net,
        const std::vector<std::vector<float>>& inputs,
        const std::vector<std::vector<float>>& targets,
        const std::vector<std::vector<float>>& val_inputs,
        const std::vector<std::vector<float>>& val_targets,
        const Config& config);

  private:
    Metric metric;
    Timing timing;
    Progress progress;

    float last_mse = 0.0f;
    float last_acc = 0.0f;

    float val_mse = 0.0f;
    float val_acc = 0.0f;

    std::chrono::duration<float> last_eta_s{};
    std::chrono::duration<float, std::milli> last_epoch_ms{};
};

#endif
