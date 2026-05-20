#ifndef LAYER_HPP
#define LAYER_HPP

#include "Neuron.hpp"
#include <vector>

class Layer
{
	public:
	 Layer(int neurons_count, int input_size);
	 std::vector<double> forward(std::vector<double> inputs);
	 std::vector<Neuron>& getNeurons();

	private:
	 std::vector<Neuron> neurons;
	 std::vector<double> outputs;
};

#endif