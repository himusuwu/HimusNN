#include "../include/Layer.hpp"
#include <cstddef>
#include <vector>

Layer::Layer(int neurons_count, int input_size)
{
	for(size_t i = 0; i < neurons_count; i++)
	{
		neurons.push_back(Neuron(input_size));
	}
}

std::vector<double> Layer::forward(std::vector<double> inputs)
{
	outputs.clear();

	for(Neuron& neuron : neurons)
	{
		outputs.push_back(neuron.calculate(inputs));
	}

	return outputs;
}

std::vector<Neuron>& Layer::getNeurons()
{
	return neurons;
}