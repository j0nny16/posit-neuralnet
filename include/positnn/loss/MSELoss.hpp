#ifndef MSELoss_HPP
#define MSELoss_HPP

// Custom headers
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

template <class ForwardT, class BackwardT=ForwardT, typename lossT=float>
class mse_loss : public Loss<BackwardT, lossT>{
public:
	mse_loss() { }

	mse_loss(StdTensor<ForwardT> const& output, StdTensor<ForwardT> const& target, Reduction reduction=Reduction::Mean) :
		error(output.shape())
	{
		size_t const size = output.size();

		if(size != target.size())
			throw std::invalid_argument( "vectors size differ" );

		// The reported loss is summed in double. Accumulating straight into the
		// scalar this->loss stalls as soon as the running sum is large enough that
		// a single squared error falls below its ULP, which for the ~10^5 elements
		// of an image batch makes the displayed loss essentially constant and
		// unusable for judging whether training is progressing.
		//
		// Only the summation changes. The per-element arithmetic stays in ForwardT,
		// so nothing about the posit evaluation is hidden, and training is
		// unaffected either way because the gradient comes from derivative() and
		// never from this scalar.
		double acc = 0.0;
		for(size_t i=0; i<size; i++){
			// Calculate loss
			ForwardT error_forward = output[i] - target[i];
			acc += static_cast<double>(error_forward * error_forward);

			// Copy to be used in backward
			error[i] = error_forward;
		}

		if(reduction == Reduction::Mean)
			acc /= static_cast<double>(size);

		this->loss = lossT(acc);
	}

	StdTensor<BackwardT> derivative() {
		StdTensor<BackwardT> dloss = error * 2;

		/*
		for(size_t i=0, size=dloss.size(); i<size; i++){
			// Calculate derivative of loss
			dloss[i] = output[i] - target[i];
			dloss[i] *= 2;
		}
		*/

		return dloss;
	}

private:
	StdTensor<BackwardT> error;
};

#endif /* MSELoss_HPP */
