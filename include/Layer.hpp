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
        float leaky_alpha,
        float elu_alpha,
        float huber_delta
    );
    Matrix forward(const Matrix& X);
    Matrix backward(const Matrix& dA, float learning_rate, float beta1, float beta2, size_t batch_size);
    Matrix backward_from_dZ(const Matrix& dZ, float learning_rate, float beta1, float beta2, size_t batch_size);

  private:
    Matrix W;           // out x in
    Matrix b;           // 1 x out

    Matrix last_input;  // batch x in
    Matrix last_output; // batch x out

    Matrix grad_W;      // out x in
    Matrix grad_b;      // 1 x out

    // Adam
    Matrix m_W; // first moment weights
    Matrix v_W; // second moment weights
    Matrix m_b; // first moment bias
    Matrix v_b; // second moment bias
    size_t t;

    ActivationType activation;
    float leaky_alpha;
    float elu_alpha;
    float huber_delta;

    float activate(float x) const;
    float activate_derivative(float y) const;
};

#endif
