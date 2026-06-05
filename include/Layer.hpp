#ifndef LAYER_HPP
#define LAYER_HPP

#include "Config.hpp"
#include "Matrix.hpp"

class Layer
{
  public:
    Layer(
        int neurons_count,
        int input_size,
        ActivationType activation,
        double leaky_alpha,
        double elu_alpha,
        double huber_delta
    );
    Matrix forward(const Matrix& X);
    Matrix backward(const Matrix& dA, double learning_rate, double momentum, size_t batch_size);
    Matrix backward_from_dZ(const Matrix& dZ, double learning_rate, double momentum, size_t batch_size);

  private:
    Matrix W;           // out x in
    Matrix b;           // 1 x out

    Matrix last_input;  // batch x in
    Matrix last_output; // batch x out

    Matrix grad_W;      // out x in
    Matrix grad_b;      // 1 x out
    Matrix velocity_W;  // out x in
    Matrix velocity_b;  // 1 x out

    ActivationType activation;
    double leaky_alpha;
    double elu_alpha;
    double huber_delta;

    double activate(double x) const;
    double activate_derivative(double y) const;
};

#endif
