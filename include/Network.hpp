#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "Layer.hpp"
#include <vector>

class Network
{
	public:
	 Network(int input_size, std::vector<int> layers_sizes, double learning_rate);
	 std::vector<double> predict(std::vector<double> inputs);
	 void train(std::vector<double> inputs, std::vector<double> targets);

	private:
	 std::vector<Layer> layers;
	 double learning_rate;
};

#endif