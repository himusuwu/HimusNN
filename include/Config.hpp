#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <vector>

struct Config 
{
	int epochs;
	int log_interval;
	int metric_interval;
	int status_interval;

	size_t bar_width;
	size_t window_size;

	bool use_color;

	double learning_rate;

	std::vector<int> layers;
};

#endif