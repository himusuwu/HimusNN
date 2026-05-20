#include "../include/Neuron.hpp"
#include <random>
#include <vector>

Neuron::Neuron(int input_size)
{
	std::mt19937 rng(std::random_device{}());
	std::uniform_real_distribution<double> dist(-0.5, 0.5);

	for(int i = 0; i < input_size; i++)
	{
		weights.push_back(dist(rng));
	}

	bias = dist(rng);
	last_output = 0.0;
}

// double sum = (w1 * x1) + (w2 * x2) + bias;

double Neuron::calculate(std::vector<double> inputs)
{
	last_inputs.clear();

	for(double& input : inputs)
	{
		last_inputs.push_back(input);
	}

	double sum = 0.0;

	if(inputs.size() == weights.size())
	{
		for(size_t i = 0; i < weights.size(); i++)
		{
			sum += weights[i] * inputs[i];
		}

		sum += bias;
	}

	last_output = activation(sum);

	return activation(sum);
}

double Neuron::calculate_out_delta(double target, double output)
{
	delta = (target - output) * derivative(output);

	return delta;
}

double Neuron::calculate_hidden_delta(double sum)
{
	delta = sum * derivative(last_output);

	return delta;
}

const std::vector<double>& Neuron::get_weights()
{
	return weights;
}

double Neuron::get_last_output()
{
	return last_output;
}

std::vector<double> Neuron::get_last_inputs()
{
	return last_inputs;
}

double Neuron::get_delta()
{
	return delta;
}

void Neuron::update_weights(double delta, double learning_rate)
{
	for(int i = 0; i < weights.size(); i++)
	{
		weights[i] += learning_rate * delta * last_inputs[i];
	}
}

void Neuron::update_bias(double delta, double learning_rate)
{
	bias += learning_rate * delta;
}