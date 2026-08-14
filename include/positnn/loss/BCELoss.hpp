#ifndef BCELOSS_HPP
#define BCELOSS_HPP

// ============================================================================
//  Binary cross entropy, in two variants matching PyTorch:
//
//    bce_loss              <-> nn.BCELoss            (input: probabilities)
//    bce_with_logits_loss  <-> nn.BCEWithLogitsLoss  (input: raw logits)
//
//  For posits the difference between the two matters more than it does in
//  float32:
//
//  * bce_loss needs log(p) and log(1-p). If the preceding sigmoid saturates at
//    exactly 1, which happens quickly with posit<8,es>, then log(1-p) = log(0)
//    -> NaR. The same holds in the backward pass, where the denominator p*(1-p)
//    becomes 0. Both places are therefore clamped, as PyTorch does with
//    eps=1e-12.
//
//  * bce_with_logits_loss works directly on the logits:
//        loss = max(x,0) - x*t + log(1 + exp(-|x|))
//    The argument of the exponential is always <= 0, so it can only underflow,
//    never overflow, and the derivative is simply sigmoid(x) - t with no
//    division at all. That is the robust choice for low-precision posits.
//
//  As in MSELoss.hpp, the reported loss is accumulated in double. Accumulating
//  into a scalar posit stalls once the running sum is large relative to a
//  single term, leaving a value that no longer tracks the error. Training is
//  unaffected either way, since the gradient comes from derivative() and never
//  from this scalar.
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
//  nn.BCELoss -- input are probabilities p in [0,1]
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

			// PyTorch clamps log() at -100 so that log(0) does not drag the loss
			// to -inf and make the reported value useless.
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

		// dL/dp = (p - t) / (p*(1-p)). The denominator is clamped from below: for
		// a saturated p (exactly 0 or 1) it would be 0 and the division would give
		// NaR. PyTorch clamps the same way with eps=1e-12, but in low-precision
		// posits that value already sits close to minpos, so the gradient
		// saturates at maxpos instead. Use bce_with_logits_loss to avoid this.
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
//  nn.BCEWithLogitsLoss -- input are raw logits x, with no sigmoid in front
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

			// sigmoid(x) as a case split, so the argument of exp() is always <= 0
			// and can only underflow, never overflow. Otherwise the double fallback
			// of exp() returns inf, and inf -> posit is NaR (see Sigmoid.hpp).
			ForwardT s;
			if(x.isneg()) {
				ForwardT const ex = exp(x);
				s = ex / (one + ex);
			}
			else {
				s = one / (one + exp(-x));
			}
			sigma[i] = s;   // kept for the backward pass

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
		// dL/dx = sigmoid(x) - t   (no division, no clamping needed)
		StdTensor<BackwardT> dloss(sigma.shape());

		for(size_t i=0, size=dloss.size(); i<size; i++)
			dloss[i] = BackwardT(sigma[i] - t[i]);

		if(m_reduction == Reduction::Mean && m_size > 0)
			dloss /= BackwardT(m_size);

		return dloss;
	}

	// sigmoid(x). The probabilities are produced by the forward pass anyway and
	// are needed for diagnostics such as D(x) and D(G(z)).
	StdTensor<ForwardT> const& probabilities() const { return sigma; }

private:
	StdTensor<ForwardT> sigma;
	StdTensor<ForwardT> t;
	Reduction m_reduction;
	size_t m_size;
};

#endif /* BCELOSS_HPP */
