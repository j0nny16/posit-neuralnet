#ifndef SIGMOID_HPP
#define SIGMOID_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
#include "../tensor/StdTensor.hpp"
#include "../utils/utils.hpp"

// Namespaces
using namespace sw::unum;

template <typename ForwardT, typename BackwardT=ForwardT>
class Sigmoid {
public:	
	Sigmoid() { }

	StdTensor<ForwardT> forward(StdTensor<ForwardT> const& x, bool approximate=true) {
		StdTensor<ForwardT> y(x.shape());

		for(size_t i=0, size=x.size(); i<size; i++) {
			if(approximate && ForwardT::es==0)
				y[i] = sigmoid_approx(x[i]);
			else {
				// Fallunterscheidung (statt 1/(1+exp(-x))): vermeidet exp(grosser
				// positiver Wert), der in der double-Fallback-exp() ueberlaufen und
				// zu NaR konvertieren kann. Siehe posit16_1_nar_analysis.md Abschnitt 11.
				if (x[i] >= ForwardT(0)) {
					y[i] = ForwardT(1) / (ForwardT(1) + exp(-x[i]));
				} else {
					auto exp_x = exp(x[i]);
					y[i] = exp_x / (ForwardT(1) + exp_x);
				}
			}
		}

		output = y;

		return y;
	}

	StdTensor<BackwardT> backward(StdTensor<BackwardT> const& w_delta) {
		StdTensor<BackwardT> deltaN = derivative();
		deltaN *= w_delta;
		return deltaN;
	}

	StdTensor<BackwardT> derivative() const {
		// TODO: protect against initialized output
		StdTensor<BackwardT> dx(output.shape());

		BackwardT pOne(1);

		for(size_t i=0, size=dx.size(); i<size; i++) {
			convert( fam_corrected(pOne, -output[i], output[i]) ,
					 dx[i] );
		}
		
		return dx;
	}

private:
	StdTensor<BackwardT> output;
};

#endif /* SIGMOID_HPP */