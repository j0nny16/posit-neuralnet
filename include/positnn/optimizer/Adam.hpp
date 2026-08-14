#ifndef ADAM_HPP
#define ADAM_HPP

#include "../layer/Parameter.hpp"
#include "../optimizer/Optimizer.hpp"
#include "../tensor/StdTensor.hpp"
#include "../tensor/matrix.hpp"      // fused()

using namespace sw::unum;

template <typename T>
struct AdamOptions {
    AdamOptions(double _learning_rate, double _beta_1=0.9, double _beta_2=0.999, double _eps=1e-8) :
        learning_rate(T(_learning_rate)),
        beta_1(T(_beta_1)),
        beta_2(T(_beta_2)),
        eps(T(_eps)),
        t(0)
    { }

    T learning_rate;
    T beta_1;
    T beta_2;
    T eps;
    size_t t;
};

template <typename T>
class Adam : public Optimizer<T> {
public:
    Adam(std::vector<Parameter<T>> parameters0, AdamOptions<T> options0) :
        Optimizer<T>(parameters0),
        _options(options0),
        _one_minus_beta1(T(1) - options0.beta_1),
        _one_minus_beta2(T(1) - options0.beta_2),
        _beta1_t(T(1)),
        _beta2_t(T(1)),
        _bias_corr1(T(1)),
        _bias_corr2(T(1))
    {
        for (auto const& p : this->_parameters) {
            _m.push_back(StdTensor<T>(p.weight.shape()));
            _v.push_back(StdTensor<T>(p.weight.shape()));
            _m.back().clear();
            _v.back().clear();
        }
    }

    void step() {
        // The bias correction only depends on t, so it is computed once per step
        // instead of once per parameter. beta^t is carried forward incrementally,
        // which is one multiplication per step instead of a pow() in O(t).
        _options.t++;
        _beta1_t *= _options.beta_1;
        _beta2_t *= _options.beta_2;
        T const one(1);
        _bias_corr1 = one / (one - _beta1_t);
        _bias_corr2 = one / (one - _beta2_t);

        Optimizer<T>::step();
    }

    AdamOptions<T>& options() { return _options; }

    std::vector<StdTensor<T>> const& moments_m() const { return _m; }
    std::vector<StdTensor<T>> const& moments_v() const { return _v; }

private:
    void update_parameter(Parameter<T>& p, size_t const i) override {
        StdTensor<T>& w = p.weight;
        StdTensor<T>& m = _m[i];
        StdTensor<T>& v = _v[i];

        StdTensor<T> const& g = p.gradient;

        // The moment updates go through fused() (matrix.hpp), so each product is
        // accumulated exactly in the quire and rounded once, instead of the three
        // to four roundings of a naive element-wise evaluation. This matches SGD,
        // which already uses fused() throughout.
        fused(m, g, _options.beta_1, _one_minus_beta1);     // m = m*b1 + g*(1-b1)
        StdTensor<T> g2 = g * g;
        fused(v, g2, _options.beta_2, _one_minus_beta2);    // v = v*b2 + g2*(1-b2)

        // The weight update needs sqrt and division, so it stays element-wise
        T const lr  = _options.learning_rate;
        T const eps = _options.eps;
        using sw::unum::sqrt;
        for (size_t j = 0, n = w.size(); j < n; ++j) {
            T m_hat = m[j] * _bias_corr1;
            T v_hat = v[j] * _bias_corr2;
            w[j] = w[j] - (lr * m_hat) / (sqrt(v_hat) + eps);
        }

        p.update();
    }

    AdamOptions<T> _options;
    std::vector<StdTensor<T>> _m;
    std::vector<StdTensor<T>> _v;
    T _one_minus_beta1, _one_minus_beta2;
    T _beta1_t, _beta2_t;                   // beta^t
    T _bias_corr1, _bias_corr2;             // written by step(), read by update_parameter()
};

#endif /* ADAM_HPP */
