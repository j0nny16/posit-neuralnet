#ifndef TANH_HPP
#define TANH_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
#include "../tensor/StdTensor.hpp"
#include "../utils/utils.hpp"

// Namespaces
using namespace sw::unum;

template <typename ForwardT, typename BackwardT=ForwardT>
class Tanh {
public:	
	Tanh() { }

	StdTensor<ForwardT> forward(StdTensor<ForwardT> const& x, bool approximate=true) {
		StdTensor<ForwardT> y(x.shape());
		
		for(size_t i=0, size=x.size(); i<size; i++) {
			if(approximate && ForwardT::es==0) {
				y[i] = tanh_approx(x[i]);
			}
			else {
				// Fallunterscheidung statt (e^x - e^-x)/(e^x + e^-x): dort geht
				// bei grossem |x| genau einer der beiden Aufrufe gegen inf. Die
				// double-Fallback-exp() liefert dann inf, und inf -> Posit ist
				// NaR. Es ist dieselbe Ueberlaufklasse, die im Sigmoid schon zum
				// NaR-Absturz gefuehrt hat (siehe Sigmoid.hpp).
				//
				//   x >= 0:  tanh(x) = (1 - e^-2x) / (1 + e^-2x)
				//   x <  0:  tanh(x) = (e^2x - 1) / (e^2x + 1)
				//
				// Das Argument von exp() ist so immer <= 0 und kann nur
				// unterlaufen — das Ergebnis saettigt korrekt gegen +-1.
				ForwardT const two(2);
				if(x[i].isneg()) {
					ForwardT const e = exp(two*x[i]);
					y[i] = (e - ForwardT(1)) / (e + ForwardT(1));
				}
				else {
					ForwardT const e = exp(-two*x[i]);
					y[i] = (ForwardT(1) - e) / (ForwardT(1) + e);
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
			convert( fma(output[i], -output[i], pOne) ,
					 dx[i] );
		}
		
		return dx;
	}

private:
	StdTensor<BackwardT> output;
};

#endif /* TANH_HPP */
