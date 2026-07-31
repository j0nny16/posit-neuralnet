#ifndef BCELOSS_HPP
#define BCELOSS_HPP

// ============================================================================
//  Binary Cross Entropy — zwei Varianten, analog zu PyTorch:
//
//    bce_loss              <-> nn.BCELoss                (Eingabe: Wahrscheinlichkeiten)
//    bce_with_logits_loss  <-> nn.BCEWithLogitsLoss      (Eingabe: rohe Logits)
//
//  Fuer Posits ist der Unterschied groesser als in float32:
//
//  * bce_loss braucht log(p) und log(1-p). Saettigt das vorgeschaltete Sigmoid
//    auf exakt 1 (bei posit<8,es> passiert das schnell), ist log(1-p) = log(0)
//    -> NaR. Genauso im Backward: der Nenner p*(1-p) wird 0. Beide Stellen sind
//    deshalb geklemmt (wie in PyTorch, dort mit eps=1e-12).
//
//  * bce_with_logits_loss rechnet direkt auf den Logits:
//        loss = max(x,0) - x*t + log(1 + exp(-|x|))
//    Das Argument der Exponentialfunktion ist immer <= 0, kann also nur
//    unterlaufen (harmlos), nie ueberlaufen. Die Ableitung ist schlicht
//    sigmoid(x) - t, ohne Division. Fuer Low-Precision-Posits ist das der
//    robuste Weg.
//
//  Der ANGEZEIGTE Loss wird — wie in MSELoss.hpp — in double aufsummiert. Wuerde
//  man in ein Skalar-Posit akkumulieren, stagniert die Summe ueber viele Terme
//  (ULP > Summand) und der Wert wird bedeutungslos. Das Training ist davon
//  unberuehrt: der Gradient kommt aus derivative(), nicht aus diesem Skalar.
// ============================================================================

// General headers
#include <cmath>
#include <stdexcept>
#include <universal/posit/posit>

// Custom headers
#include "Loss.hpp"
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

// ---------------------------------------------------------------------------
//  nn.BCELoss — Eingabe sind Wahrscheinlichkeiten p in [0,1]
// ---------------------------------------------------------------------------
template <class ForwardT, class BackwardT=ForwardT, typename lossT=float>
class bce_loss : public Loss<BackwardT, lossT> {
public:
	bce_loss() { }

	bce_loss(StdTensor<ForwardT> const& output, StdTensor<ForwardT> const& target,
			 Reduction reduction=Reduction::Mean) :
		p(output), t(target), m_reduction(reduction), m_size(output.size())
	{
		if(output.size() != target.size())
			throw std::invalid_argument("bce_loss: vectors size differ");

		double acc = 0.0;

		for(size_t i=0, size=output.size(); i<size; i++) {
			double const pi = static_cast<double>(output[i]);
			double const ti = static_cast<double>(target[i]);

			// PyTorch klemmt log() nach unten auf -100, damit log(0) den Loss
			// nicht auf -inf zieht (und damit die Anzeige unbrauchbar macht).
			double const log_p   = std::max(std::log(pi),       -100.0);
			double const log_1mp = std::max(std::log(1.0 - pi), -100.0);

			acc -= ti * log_p + (1.0 - ti) * log_1mp;
		}

		if(reduction == Reduction::Mean && m_size > 0)
			acc /= static_cast<double>(m_size);

		this->loss = lossT(acc);
	}

	StdTensor<BackwardT> derivative() override {
		StdTensor<BackwardT> dloss(p.shape());

		// dL/dp = (p - t) / (p*(1-p)).  Der Nenner wird nach unten geklemmt:
		// bei gesaettigtem p (exakt 0 oder 1) waere er 0 und die Division ergaebe
		// NaR. PyTorch benutzt dieselbe Klemmung mit eps=1e-12; in Low-Precision-
		// Posits landet dieser Wert allerdings schon nahe minpos, der Gradient
		// saettigt dann auf maxpos. Wer das vermeiden will, nimmt
		// bce_with_logits_loss.
		ForwardT const one(1);
		ForwardT const eps(1e-12);

		for(size_t i=0, size=dloss.size(); i<size; i++) {
			ForwardT denom = p[i] * (one - p[i]);
			if(denom.isneg() || denom.iszero() || denom < eps)
				denom = eps;

			dloss[i] = BackwardT( (p[i] - t[i]) / denom );
		}

		if(m_reduction == Reduction::Mean && m_size > 0)
			dloss /= BackwardT(m_size);

		return dloss;
	}

private:
	StdTensor<ForwardT> p;
	StdTensor<ForwardT> t;
	Reduction m_reduction;
	size_t m_size;
};

// ---------------------------------------------------------------------------
//  nn.BCEWithLogitsLoss — Eingabe sind rohe Logits x (kein Sigmoid davor!)
// ---------------------------------------------------------------------------
template <class ForwardT, class BackwardT=ForwardT, typename lossT=float>
class bce_with_logits_loss : public Loss<BackwardT, lossT> {
public:
	bce_with_logits_loss() { }

	bce_with_logits_loss(StdTensor<ForwardT> const& output, StdTensor<ForwardT> const& target,
						 Reduction reduction=Reduction::Mean) :
		sigma(output.shape()), t(target), m_reduction(reduction), m_size(output.size())
	{
		if(output.size() != target.size())
			throw std::invalid_argument("bce_with_logits_loss: vectors size differ");

		double acc = 0.0;
		ForwardT const one(1);

		for(size_t i=0, size=output.size(); i<size; i++) {
			ForwardT const x = output[i];

			// sigmoid(x) mit Fallunterscheidung: das Argument von exp() ist so
			// immer <= 0 und kann nur unterlaufen, nie ueberlaufen (die
			// double-Fallback-exp() wuerde sonst inf liefern -> NaR).
			ForwardT s;
			if(x.isneg()) {
				ForwardT const ex = exp(x);
				s = ex / (one + ex);
			}
			else {
				s = one / (one + exp(-x));
			}
			sigma[i] = s;   // fuer das Backward gemerkt

			// loss = max(x,0) - x*t + log(1 + exp(-|x|))
			double const xd = static_cast<double>(x);
			double const td = static_cast<double>(target[i]);
			acc += std::max(xd, 0.0) - xd * td + std::log1p(std::exp(-std::abs(xd)));
		}

		if(reduction == Reduction::Mean && m_size > 0)
			acc /= static_cast<double>(m_size);

		this->loss = lossT(acc);
	}

	StdTensor<BackwardT> derivative() override {
		// dL/dx = sigmoid(x) - t   (keine Division, keine Klemmung noetig)
		StdTensor<BackwardT> dloss(sigma.shape());

		for(size_t i=0, size=dloss.size(); i<size; i++)
			dloss[i] = BackwardT(sigma[i] - t[i]);

		if(m_reduction == Reduction::Mean && m_size > 0)
			dloss /= BackwardT(m_size);

		return dloss;
	}

	// sigmoid(x) — die Wahrscheinlichkeiten fallen im Forward ohnehin an und
	// werden fuer Diagnosen wie D(x) / D(G(z)) gebraucht.
	StdTensor<ForwardT> const& probabilities() const { return sigma; }

private:
	StdTensor<ForwardT> sigma;
	StdTensor<ForwardT> t;
	Reduction m_reduction;
	size_t m_size;
};

#endif /* BCELOSS_HPP */
