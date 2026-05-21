#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <vector>
#include <algorithm>

struct Config 
{
	int epochs = 50000;
	int log_interval;
	int metric_interval = std::max(1, epochs / 100);
	int status_interval = 10;

	size_t bar_width = 40;
	size_t window_size = 20;

	std::vector<int> layers;
	double learning_rate;

	bool use_color = true;
};

#endif