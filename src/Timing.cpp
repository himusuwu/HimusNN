#include "Timing.hpp"
#include <ratio>

Timing::Timing(size_t window_size) : window_size(window_size) {}

Timing::TimingResult Timing::update(std::chrono::steady_clock::duration epoch_time, int epochs_left)
{
	recent_times.push_front(epoch_time);
	sum_recent_times += epoch_time;

	if(recent_times.size() > window_size)
	{
		sum_recent_times -= recent_times.back();
		recent_times.pop_back();
	}

	std::chrono::duration<double> avg_epoch_time = sum_recent_times / recent_times.size();
    std::chrono::duration<double> eta = epochs_left * avg_epoch_time;

	std::chrono::duration<double, std::milli> epoch_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(epoch_time);

	return Timing::TimingResult{avg_epoch_time, eta, epoch_ms};
}