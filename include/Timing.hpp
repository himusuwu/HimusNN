#ifndef TIMING_HPP
#define TIMING_HPP

#include <chrono>
#include <deque>
#include <ratio>

class Timing
{
  public:
    Timing(size_t window_size);

    struct TimingResult
    {
        std::chrono::duration<float> avg;
        std::chrono::duration<float> eta;
        std::chrono::duration<float, std::milli> epoch_ms;
    };

    TimingResult update(std::chrono::steady_clock::duration epoch_time, int epochs_left);

  private:
    std::chrono::duration<float> sum_recent_times{};
    std::deque<std::chrono::steady_clock::duration> recent_times{};

    size_t window_size;
};

#endif
