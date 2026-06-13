#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "Config.hpp"
#include "DataSet.hpp"
#include "Metric.hpp"
#include "Network.hpp"
#include "Progress.hpp"
#include "Timing.hpp"

class Trainer
{
  public:
    explicit Trainer(const Config& config) : timing(config.window_size), progress(config.bar_width, config.use_color) {}

    void run(Network& net, const DataSet& train_data, const DataSet& val_data, const Config& config);

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
