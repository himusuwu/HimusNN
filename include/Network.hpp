#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "Config.hpp"
#include "Layer.hpp"
#include <vector>

class Network
{
	public:
	 Network(int input_size, std::vector<int> layers_sizes, double learning_rate, ActivationType activation, LossType loss, double leaky_alpha, double elu_alpha, double huber_delta);
	 std::vector<double> predict(std::vector<double> inputs);
	 void train(std::vector<double> inputs, std::vector<double> targets);

	private:
	 std::vector<Layer> layers;
	 double learning_rate;

	 ActivationType activation;
	 LossType loss;
};

#endif