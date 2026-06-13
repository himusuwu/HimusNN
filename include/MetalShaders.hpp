#ifndef METAL_SHADERS_HPP
#define METAL_SHADERS_HPP

#ifdef __APPLE__

constexpr const char* SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

kernel void forward_activation(
	device float* Z [[buffer(0)]],
	constant float* bias [[buffer(1)]],
	constant uint& cols [[buffer(2)]],
	constant uint& activation [[buffer(3)]],
	constant float& leaky_alpha [[buffer(4)]],
	constant float& elu_alpha [[buffer(5)]],
	uint id [[thread_position_in_grid]]
) {
	uint col = id % cols;
	
	float x = Z[id] + bias[col];
	if(isnan(x) || isinf(x))
	{
		x = 0.0;
	}

	x += bias[col];

	if (activation == 0)
	{
		x = clamp(x, -50.0, 50.0);
		x = 1.0 / (1.0 + exp(-x));
	}
	else if (activation == 1)
	{
		x = tanh(x);
	}
	else if (activation == 2)
	{
		x = max(0.0, x);
	}
	else if (activation == 3)
	{
		x = (x > 0.0) ? x : leaky_alpha * x;
	}
	else if (activation == 4)
	{
		x = (x > 0.0) ? x : elu_alpha * (exp(x) - 1.0);
	}
	else if (activation == 5)
	{
		x = clamp(x, -50.0, 50.0);
		x = log(1.0 + exp(x));
	}

	Z[id] = x;
}

kernel void adam_update(
	device float* W [[buffer(0)]],
	device float* m [[buffer(1)]],
	device float* v [[buffer(2)]],
	constant float* grad [[buffer(3)]],
	constant float& b1 [[buffer(4)]],
	constant float& b2 [[buffer(5)]],
	constant float& b1_t [[buffer(6)]],
	constant float& b2_t [[buffer(7)]],
	constant float& lr [[buffer(8)]],
	constant float& eps [[buffer(9)]],
	uint id [[thread_position_in_grid]]
) {
	float g = grad[id];
	
	float new_m = b1 * m[id] + (1.0 - b1) * g;
	float new_v = b2 * v[id] + (1.0 - b2) * g * g;

	m[id] = new_m;
	v[id] = new_v;

	float m_hat = new_m / (1.0 - b1_t);
	float v_hat = new_v / (1.0 - b2_t);

	W[id] -= lr * (m_hat / (sqrt(v_hat) + eps));
}

)";

#endif
#endif
