#ifndef BATCHNORM2D_HPP
#define BATCHNORM2D_HPP

// ============================================================================
//  BatchNorm2d  —  Batch-Normalisierung fuer 4D-Tensoren [N, C, H, W].
//
//  Statistiken werden PRO KANAL ueber N*H*W Elemente gebildet (PyTorch-Semantik).
//  Damit ist das NICHT auf BatchNorm1d zurueckfuehrbar: BatchNorm1d nimmt an,
//  dass der Feature-Index der am schnellsten laufende ist (Layout [N, F]),
//  waehrend der Kanal bei NCHW mit Schrittweite H*W laeuft.
//
//  Mixed Precision wie bei Conv2d/TransposedConv2d (OptimizerT/ForwardT/
//  BackwardT/GradientT) statt wie bei BatchNorm1d mit einem einzigen Posit-Typ.
//  Nur so bleibt BatchNorm im Format-Experiment ueberhaupt sichtbar — sonst
//  liefe ausgerechnet die Schicht, die alle Aktivierungen skaliert, in einer
//  anderen Praezision als der Rest des Forward-Pfads.
//
//  Summen laufen ueber das Quire (exakte Akkumulation, eine Rundung am Ende).
//  Das ist hier besonders wichtig: die Kanal-Summen gehen bei einem
//  GAN-Discriminator ueber N*H*W = 64*7*7 = 3136 Summanden.
//
//  Formeln (Training):
//      mean[c]  = 1/M * sum(x)                      M = N*H*W
//      var[c]   = 1/M * sum((x-mean)^2)             (biased, fuer die Normalisierung)
//      x_norm   = (x - mean) / sqrt(var + eps)
//      y        = gamma[c] * x_norm + beta[c]
//      running_mean = (1-momentum)*running_mean + momentum*mean
//      running_var  = (1-momentum)*running_var  + momentum*var*M/(M-1)   (unbiased)
//
//  Backward (delta = dL/dy):
//      dbeta[c]  = sum(delta)
//      dgamma[c] = sum(delta * x_norm)
//      dx = gamma[c] / (M*stddev[c]) * (M*delta - dbeta[c] - x_norm*dgamma[c])
//
//  Die Parameter-Gradienten werden NICHT durch die Batchgroesse geteilt — genau
//  wie in Conv2d dieses Repos ("Divisionen im Gradient werden ausgeschalten,
//  das uebernimmt der Loss"). PyTorch macht es genauso: der Faktor 1/N steckt
//  bereits im delta der gemittelten Loss-Funktion.
// ============================================================================

// General headers
#include <cmath>
#include <stdexcept>
#include <universal/posit/posit>

// Custom headers
#include "Layer.hpp"
#include "../tensor/MixedTensor.hpp"
#include "../tensor/StdTensor.hpp"
#include "../utils/Quire.hpp"

// Namespaces
using namespace sw::unum;

template <typename OptimizerT, typename ForwardT=OptimizerT, typename BackwardT=ForwardT, typename GradientT=BackwardT>
class BatchNorm2d : public Layer<OptimizerT> {
public:
	BatchNorm2d(size_t _num_features, double _eps=1e-5, double _momentum=0.1,
				bool _affine=true, bool _track_running_stats=true) :
		num_features(_num_features),
		eps(_eps), momentum(_momentum),
		affine(_affine), track_running_stats(_track_running_stats),
		gamma(_num_features), beta(_num_features),
		gamma_gradient(_num_features), beta_gradient(_num_features),
		running_mean(_num_features), running_variance(_num_features),
		mean(_num_features), stddev(_num_features)
	{
		this->register_parameter(gamma, gamma_gradient);
		this->register_parameter(beta, beta_gradient);

		reset_parameters();
	}

	void reset_parameters() {
		// PyTorch-Init: gamma=1, beta=0, running_mean=0, running_var=1
		gamma.get_optimizer().set(OptimizerT(1));
		beta.get_optimizer().set(OptimizerT(0));
		gamma.update();
		beta.update();

		running_mean.set(ForwardT(0));
		running_variance.set(ForwardT(1));
	}

	// -----------------------------------------------------------------------
	//  Forward
	// -----------------------------------------------------------------------
	StdTensor<ForwardT> forward(StdTensor<ForwardT> const& x) {
		if(x.dim() != 4)
			throw std::invalid_argument("BatchNorm2d expects a 4D tensor [N,C,H,W]");
		if(x.shape()[1] != num_features)
			throw std::invalid_argument("BatchNorm2d: channel count does not match num_features");

		batch      = x.shape()[0];
		spatial    = x.shape()[2] * x.shape()[3];
		size_t const M = batch * spatial;

		StdTensor<ForwardT> y(x.shape());

		if(Layer<OptimizerT>::training || !track_running_stats) {
			StdTensor<ForwardT> variance(num_features);
			calculate_mean_variance(x, mean, variance, M);

			for(size_t c=0; c<num_features; c++)
				stddev[c] = sqrt(variance[c] + ForwardT(eps));

			if(track_running_stats)
				update_running_stats(mean, variance, M);

			normalize(x, y, mean, stddev);

			// x_norm wird im Backward gebraucht -> in BackwardT sichern
			x_norm = StdTensor<BackwardT>(y);
		}
		else {
			StdTensor<ForwardT> eval_stddev(num_features);
			for(size_t c=0; c<num_features; c++)
				eval_stddev[c] = sqrt(running_variance[c] + ForwardT(eps));

			normalize(x, y, running_mean, eval_stddev);
		}

		if(affine) {
			StdTensor<ForwardT>& g = gamma.get_forward();
			StdTensor<ForwardT>& b = beta.get_forward();
			value<1 + 2 * (ForwardT::nbits - ForwardT::es)> result;

			for(size_t i=0, size=y.size(); i<size; i++) {
				size_t const c = channel_of(i);
				// gamma*x_norm + beta als fused multiply-add: eine Rundung statt zwei
				result = fma(y[i], g[c], b[c]);
				convert(result, y[i]);
			}
		}

		return y;
	}

	// -----------------------------------------------------------------------
	//  Backward
	// -----------------------------------------------------------------------
	StdTensor<BackwardT> backward(StdTensor<BackwardT> const& delta) {
		size_t const M = batch * spatial;

		// Kanal-Summen: sum(delta) und sum(delta*x_norm). Beides sind exakt die
		// Gradienten von beta und gamma, sie werden deshalb nur EINMAL berechnet
		// und danach fuer den Input-Gradienten wiederverwendet.
		StdTensor<BackwardT> sum_delta(num_features);
		StdTensor<BackwardT> sum_delta_xnorm(num_features);
		channel_sums(delta, sum_delta, sum_delta_xnorm);

		if(affine)
			accumulate_gradients(sum_delta, sum_delta_xnorm);

		StdTensor<BackwardT>& g = gamma.get_backward();
		StdTensor<BackwardT> dx(delta.shape());

		// Vorfaktor pro Kanal: gamma / (M * stddev)
		StdTensor<BackwardT> factor(num_features);
		for(size_t c=0; c<num_features; c++) {
			BackwardT const denom = BackwardT(stddev[c]) * BackwardT(M);
			factor[c] = (affine ? g[c] : BackwardT(1)) / denom;
		}

		BackwardT const pM(M);
		Quire<BackwardT::nbits, BackwardT::es> q;

		for(size_t i=0, size=delta.size(); i<size; i++) {
			size_t const c = channel_of(i);

			// M*delta - sum_delta - x_norm*sum_delta_xnorm  (exakt im Quire)
			q = Quire_mul(delta[i], pM);
			q -= sum_delta[c];
			q -= Quire_mul(x_norm[i], sum_delta_xnorm[c]);
			convert(q.to_value(), dx[i]);

			dx[i] *= factor[c];
		}

		return dx;
	}

	// -----------------------------------------------------------------------
	//  Speichern/Laden: zusaetzlich zu den Parametern auch die laufenden Statistiken
	// -----------------------------------------------------------------------
	template <typename PositFile=OptimizerT>
	void write(std::ostream& out) {
		Layer<OptimizerT>::template write<PositFile>(out);
		running_mean.template write<PositFile>(out);
		running_variance.template write<PositFile>(out);
	}

	template <typename PositFile=OptimizerT>
	void read(std::istream& in) {
		Layer<OptimizerT>::template read<PositFile>(in);
		running_mean.template read<PositFile>(in);
		running_variance.template read<PositFile>(in);
	}

	// Nur die laufenden Statistiken — ohne gamma/beta.
	//
	// Wird gebraucht, wenn die Schicht Teil eines groesseren Netzes ist: dessen
	// Layer::write schreibt zwar ALLE registrierten Parameter (also auch
	// gamma/beta dieser Schicht), kennt running_mean/running_variance aber nicht.
	// Die sind keine Parameter, sondern Buffer — und `write` ist ein Template und
	// kann deshalb nicht virtual sein, das Netz kann den Aufruf also nicht an die
	// Schicht durchreichen. Ohne diese beiden Aufrufe liefert ein wieder
	// eingelesenes Netz im eval()-Modus andere Werte als vor dem Speichern.
	template <typename PositFile=OptimizerT>
	void write_running(std::ostream& out) {
		running_mean.template write<PositFile>(out);
		running_variance.template write<PositFile>(out);
	}

	template <typename PositFile=OptimizerT>
	void read_running(std::istream& in) {
		running_mean.template read<PositFile>(in);
		running_variance.template read<PositFile>(in);
	}

	// Zugriff fuer Tests/Debug bzw. zum Uebernehmen von PyTorch-Buffern.
	StdTensor<ForwardT>& get_running_mean()     { return running_mean; }
	StdTensor<ForwardT>& get_running_variance() { return running_variance; }

private:
	// Kanal eines flachen Index bei Layout [N, C, H, W]
	inline size_t channel_of(size_t i) const {
		return (i / spatial) % num_features;
	}

	void calculate_mean_variance(StdTensor<ForwardT> const& x,
								 StdTensor<ForwardT>& mean_out,
								 StdTensor<ForwardT>& variance_out,
								 size_t const M) {
		constexpr size_t nbits = ForwardT::nbits;
		constexpr size_t es    = ForwardT::es;

		ForwardT const pM(M);
		Quire<nbits, es> q;

		for(size_t c=0; c<num_features; c++) {
			// Mittelwert
			q.clear();
			for(size_t n=0; n<batch; n++) {
				size_t const base = (n*num_features + c) * spatial;
				for(size_t k=0; k<spatial; k++)
					q += x[base+k];
			}
			convert(q.to_value(), mean_out[c]);
			mean_out[c] /= pM;

			// Varianz (biased) — zweiter Durchlauf statt E[x^2]-E[x]^2, weil die
			// Differenzform bei grossen Mittelwerten katastrophal ausloeschen kann.
			q.clear();
			for(size_t n=0; n<batch; n++) {
				size_t const base = (n*num_features + c) * spatial;
				for(size_t k=0; k<spatial; k++) {
					ForwardT const d = x[base+k] - mean_out[c];
					q += Quire_mul(d, d);
				}
			}
			convert(q.to_value(), variance_out[c]);
			variance_out[c] /= pM;
		}
	}

	void normalize(StdTensor<ForwardT> const& x, StdTensor<ForwardT>& y,
				   StdTensor<ForwardT> const& mean_in, StdTensor<ForwardT> const& stddev_in) {
		for(size_t i=0, size=x.size(); i<size; i++) {
			size_t const c = channel_of(i);
			y[i] = (x[i] - mean_in[c]) / stddev_in[c];
		}
	}

	void update_running_stats(StdTensor<ForwardT> const& mean_in,
							  StdTensor<ForwardT> const& variance_in,
							  size_t const M) {
		ForwardT const mom(momentum);
		ForwardT const one_minus_mom = ForwardT(1) - mom;
		// PyTorch fuehrt running_var mit der ERWARTUNGSTREUEN Varianz nach
		// (Bessel-Korrektur M/(M-1)), normalisiert im Training aber mit der
		// biased-Variante. Ohne diese Unterscheidung weicht der eval()-Pfad
		// systematisch von PyTorch ab.
		ForwardT const bessel = (M > 1) ? ForwardT(double(M) / double(M-1)) : ForwardT(1);

		value<1 + 2 * (ForwardT::nbits - ForwardT::es)> result;
		for(size_t c=0; c<num_features; c++) {
			result = fma(running_mean[c], one_minus_mom, ForwardT(mean_in[c]*mom));
			convert(result, running_mean[c]);

			result = fma(running_variance[c], one_minus_mom, ForwardT(variance_in[c]*bessel*mom));
			convert(result, running_variance[c]);
		}
	}

	void channel_sums(StdTensor<BackwardT> const& delta,
					  StdTensor<BackwardT>& sum_delta,
					  StdTensor<BackwardT>& sum_delta_xnorm) {
		constexpr size_t nbits = BackwardT::nbits;
		constexpr size_t es    = BackwardT::es;

		Quire<nbits, es> q_sum, q_dot;

		for(size_t c=0; c<num_features; c++) {
			q_sum.clear();
			q_dot.clear();
			for(size_t n=0; n<batch; n++) {
				size_t const base = (n*num_features + c) * spatial;
				for(size_t k=0; k<spatial; k++) {
					q_sum += delta[base+k];
					q_dot += Quire_mul(delta[base+k], x_norm[base+k]);
				}
			}
			convert(q_sum.to_value(), sum_delta[c]);
			convert(q_dot.to_value(), sum_delta_xnorm[c]);
		}
	}

	void accumulate_gradients(StdTensor<BackwardT> const& sum_delta,
							  StdTensor<BackwardT> const& sum_delta_xnorm) {
		for(size_t c=0; c<num_features; c++) {
			beta_gradient[c]  += OptimizerT(sum_delta[c]);
			gamma_gradient[c] += OptimizerT(sum_delta_xnorm[c]);
		}
	}

	size_t const num_features;
	double eps;
	double momentum;
	bool affine;
	bool track_running_stats;

	MixedTensor<OptimizerT, ForwardT, BackwardT> gamma;
	MixedTensor<OptimizerT, ForwardT, BackwardT> beta;
	StdTensor<OptimizerT> gamma_gradient;
	StdTensor<OptimizerT> beta_gradient;

	StdTensor<ForwardT> running_mean;
	StdTensor<ForwardT> running_variance;

	// Zwischenwerte aus dem Forward, die das Backward braucht
	StdTensor<ForwardT>  mean;
	StdTensor<ForwardT>  stddev;
	StdTensor<BackwardT> x_norm;
	size_t batch   = 0;
	size_t spatial = 0;
};

#endif /* BATCHNORM2D_HPP */
