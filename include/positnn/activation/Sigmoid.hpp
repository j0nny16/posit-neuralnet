#ifndef SIGMOID_HPP
#define SIGMOID_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
#include "../tensor/StdTensor.hpp"
#include "../utils/utils.hpp"

#include "../utils/health_utils.hpp"
#include <iostream>

// Namespaces
using namespace sw::unum;

template <typename ForwardT, typename BackwardT=ForwardT>
class Sigmoid {
public:	
	Sigmoid() { }

	StdTensor<ForwardT> forward(StdTensor<ForwardT> const& x, bool approximate=true, bool clamp=false) {
		StdTensor<ForwardT> y(x.shape());

		constexpr size_t nbits = ForwardT::nbits;
        constexpr size_t es = ForwardT::es;
        
        // Formel: max_x = (2^es) * (nbits - 2) * ln(2)
        // Wir nutzen (1 << es) für 2^es und ziehen 0.1 als Sicherheits-Puffer ab
        double const max_safe_x = (double(1 << es) * double(nbits - 2) * 0.6931471805599453) - 0.1;
        ForwardT const clamp_limit(max_safe_x);

		for(size_t i=0, size=x.size(); i<size; i++) {
			if (x[i].isnar()) {
				std::cerr << "[Sigmoid NaR]: NaR vor Sigmoid Funktion gefunden! " << "\n";
				inspect_tensor_detailed(x);
                y[i] = ForwardT(0.5); // Sicherer Fallback (Mittelpunkt der Sigmoid)
                continue;
            }

			ForwardT x_clamped = x[i];
            if (clamp && x_clamped > clamp_limit)
			{
				//std::cerr << "[Sigmoid Clamp]: Index " << i << " over limit (" << float(x_clamped) << "). Clamping to " << float(clamp_limit) << "\n";
				x_clamped = clamp_limit;
			}  
            if (clamp && x_clamped < -clamp_limit)
			{
				//std::cerr << "[Sigmoid Clamp]: Index " << i << " under limit (" << float(x_clamped) << "). Clamping to " << float(-clamp_limit) << "\n";
				x_clamped = -clamp_limit;
			} 

			if(approximate && ForwardT::es==0)
				y[i] = sigmoid_approx(x[i]);
			else
			{
				//y[i] = 1/(1+exp(-x[i]));
				if (x[i] >= ForwardT(0)) {
					y[i] = ForwardT(1) / (ForwardT(1) + exp(-x[i]));
				} else {
					auto exp_x = exp(x[i]);
					y[i] = exp_x / (ForwardT(1) + exp_x);
				}

			}

			if (y[i].isnar()) {
                std::cerr << "[SIGMOID CRITICAL] Generated NaR at index " << i << " for input " << (double)x[i] << "! Safe-clamped to 0.0\n";
                y[i] = ForwardT(0);
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
