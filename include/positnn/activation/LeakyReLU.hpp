#ifndef LEAKYRELU_HPP
#define LEAKYRELU_HPP

// ============================================================================
//  LeakyReLU  --  f(x) = x           for x > 0
//                 f(x) = slope * x   otherwise
//
//  Deliberately structured like ReLU.hpp (a stateless functor that remembers a
//  mask), so the two can be swapped in a network without further changes.
//
//  The mask stores "x <= 0" rather than "x < 0". PyTorch's backward uses the
//  condition x > 0 as well, so x == 0 belongs to the negative branch and gets
//  the slope. The forward value is 0 either way; only the gradient differs, and
//  it should match PyTorch.
//
//  Unlike ReLU the negative branch is scaled instead of zeroed, which matters
//  for posits: slope*x can land below minpos. With UNDERFLOW_MODE=0 (the
//  default here) the result saturates at minpos instead of flushing to zero,
//  so LeakyReLU does not create the dead range that ReLU has.
// ============================================================================

// General headers
#include <universal/posit/posit>
#include <vector>

// Custom headers
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

class LeakyReLU {
public:
	LeakyReLU(float _negative_slope=0.01) :
		negative_slope(_negative_slope)
	{ }

	template <typename T>
	StdTensor<T> forward(StdTensor<T> x) {
		negative.resize(x.size());
		T const slope(negative_slope);

		for(size_t i=0, size=x.size(); i<size; i++) {
			if(x[i].isneg() || x[i].iszero()) {
				x[i] *= slope;
				negative[i] = true;
			}
			else {
				negative[i] = false;
			}
		}

		return x;
	}

	template <typename T>
	StdTensor<T> backward(StdTensor<T> delta) {
		T const slope(negative_slope);

		for(size_t i=0, size=delta.size(); i<size; i++) {
			if(negative[i]) {
				delta[i] *= slope;
			}
		}

		return delta;
	}

	float slope() const { return negative_slope; }

private:
	float negative_slope;
	std::vector<bool> negative;
};

#endif /* LEAKYRELU_HPP */
