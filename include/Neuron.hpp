#ifndef NEURON_HPP
#define NEURON_HPP

#include <cmath>
#include <vector>

class Neuron
{
	public:
	Neuron(int inputs_size);  
	double calculate(std::vector<double> inputs);
	double calculate_out_delta(double target, double output);
	double calculate_hidden_delta(double sum);
	const std::vector<double>& get_weights();
	double get_last_output();
	std::vector<double> get_last_inputs();
	double get_delta();
	void update_weights(double delta, double learning_rate);
	void update_bias(double delta, double learning_rate);

	private:
	  std::vector<double> weights;
	  double bias;
	  double last_output;
	  std::vector<double> last_inputs;
	  double delta;

	  double activation(double sum)
	  {
		return (1 / (1 + exp(-sum)));
	  }

	  double derivative(double last_output)
	  {
		return (last_output * (1 - last_output));
	  }
};

#endif