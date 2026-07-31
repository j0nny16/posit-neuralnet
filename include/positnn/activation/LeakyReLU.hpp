#ifndef LEAKYRELU_HPP
#define LEAKYRELU_HPP

// ============================================================================
//  LeakyReLU  —  f(x) = x           fuer x > 0
//                f(x) = slope * x   sonst
//
//  Aufbau bewusst identisch zu ReLU.hpp (zustandsloser Funktor mit gemerkter
//  Maske), damit die Layer in den Netzen austauschbar sind.
//
//  Die Maske speichert "x <= 0" statt "x < 0": PyTorch benutzt im Backward
//  ebenfalls die Bedingung x > 0 (also gehoert x == 0 zum negativen Ast und
//  bekommt slope). Der Forward-Wert ist an der Stelle in beiden Faellen 0, nur
//  der Gradient unterscheidet sich — und der soll zu PyTorch passen.
//
//  Anders als ReLU wird der negative Ast NICHT genullt, sondern skaliert. Fuer
//  Posits ist das relevant: slope*x kann in den Bereich unterhalb von minpos
//  fallen. Bei UNDERFLOW_MODE=0 (Repo-Default) saettigt das Ergebnis auf minpos
//  statt auf 0, der "tote Bereich" der ReLU entsteht hier also nicht.
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
