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

// ============================================================================
//  Adam-Optimizer.
//
//  Die Momenten-Updates nutzen fused() aus matrix.hpp:
//      m = m*beta1 + g *(1-beta1)
//      v = v*beta2 + g²*(1-beta2)
//  Beide Produkte werden im Quire exakt akkumuliert und genau EINMAL gerundet
//  (statt 3-4 Rundungen bei naiver element-weiser Rechnung). Konsistent zu SGD,
//  das ebenfalls durchgaengig fused() verwendet.
//
//  Das Gewichts-Update braucht sqrt und Division und bleibt element-weise.
//  Die Bias-Korrektur haengt nur vom globalen Schritt t ab und wird deshalb
//  EINMAL pro step() berechnet (nicht pro Layer); beta^t wird inkrementell
//  fortgeschrieben (eine Multiplikation/Schritt statt posit_pow in O(t)).
// ============================================================================
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
        // Adam braucht zwei Status-Buffer (m, v) pro Parameter, mit 0 initialisiert.
        for (auto const& p : this->_parameters) {
            _m.push_back(StdTensor<T>(p.weight.shape()));
            _v.push_back(StdTensor<T>(p.weight.shape()));
            _m.back().clear();
            _v.back().clear();
        }
    }

    void step(double loss_scale = 1.0) {
        // Ein globaler Zeitschritt pro Batch.
        _options.t++;

        // beta^t inkrementell (O(1) statt posit_pow in O(t)); Bias-Korrektur einmal
        // pro Schritt, da nur von t abhaengig (gilt fuer alle Layer gleich).
        _beta1_t *= _options.beta_1;
        _beta2_t *= _options.beta_2;
        T const one(1);
        _bias_corr1 = one / (one - _beta1_t);
        _bias_corr2 = one / (one - _beta2_t);

        Optimizer<T>::step(loss_scale);   // verteilt update_parameter auf Threads
    }

    AdamOptions<T>& options() { return _options; }

    // Lesezugriff auf die Momenten-Buffer (fuer Analyse/Tests)
    std::vector<StdTensor<T>> const& moments_m() const { return _m; }
    std::vector<StdTensor<T>> const& moments_v() const { return _v; }

    // ------------------------------------------------------------------------
    //  Serialisierung des Optimizer-Zustands (fuer Checkpoint/Resume).
    //
    //  Ohne m, v und t ist ein fortgesetzter Lauf KEIN fortgesetzter Lauf: Adam
    //  faengt mit leeren Momenten wieder bei der Bias-Korrektur von t=1 an und
    //  macht ein paar unverhaeltnismaessig grosse Schritte. Bei einem GAN reicht
    //  das, um ein austariertes Gleichgewicht zu zerstoeren.
    //
    //  Format: t, dann je Parameter m und v. Die Bias-Korrektur-Faktoren werden
    //  aus t rekonstruiert statt mitgeschrieben (haengen nur von t und den betas
    //  ab, und die kommen ohnehin aus der Config).
    // ------------------------------------------------------------------------
    template <typename PositFile=T>
    void write(std::ostream& out) {
        size_t const t = _options.t;
        out.write((char*)&t, sizeof(t));
        for (StdTensor<T>& m : _m) m.template write<PositFile>(out);
        for (StdTensor<T>& v : _v) v.template write<PositFile>(out);
    }

    template <typename PositFile=T>
    void read(std::istream& in) {
        size_t t = 0;
        in.read((char*)&t, sizeof(t));
        _options.t = t;
        for (StdTensor<T>& m : _m) m.template read<PositFile>(in);
        for (StdTensor<T>& v : _v) v.template read<PositFile>(in);

        // beta^t nachziehen, damit die Bias-Korrektur im naechsten step() stimmt
        _beta1_t = T(1);
        _beta2_t = T(1);
        for (size_t i = 0; i < t; ++i) {
            _beta1_t *= _options.beta_1;
            _beta2_t *= _options.beta_2;
        }
    }

private:
    void update_parameter(Parameter<T>& p, size_t const i, double loss_scale = 1.0) override {
        StdTensor<T>& w = p.weight;
        StdTensor<T>& m = _m[i];
        StdTensor<T>& v = _v[i];

        // Skalierte Gradienten (Loss-Scaling rueckgaengig). Bei scale==1 keine Kosten.
        StdTensor<T> g = p.gradient;
        if (loss_scale != 1.0)
            g *= T(1.0 / loss_scale);

        // Momenten-Updates: fused (Quire, je eine Rundung)
        fused(m, g, _options.beta_1, _one_minus_beta1);     // m = m*b1 + g*(1-b1)
        StdTensor<T> g2 = g * g;
        fused(v, g2, _options.beta_2, _one_minus_beta2);    // v = v*b2 + g2*(1-b2)

        // Gewichts-Update: sqrt + Division, zwingend element-weise
        T const lr  = _options.learning_rate;
        T const eps = _options.eps;
        using sw::unum::sqrt;
        for (size_t j = 0, n = w.size(); j < n; ++j) {
            T m_hat = m[j] * _bias_corr1;
            T v_hat = v[j] * _bias_corr2;
            w[j] = w[j] - (lr * m_hat) / (sqrt(v_hat) + eps);
        }

        p.update();   // Mixed-Precision: ggf. Forward/Backward-Gewichte aktualisieren
    }

    AdamOptions<T> _options;
    std::vector<StdTensor<T>> _m;
    std::vector<StdTensor<T>> _v;
    T _one_minus_beta1, _one_minus_beta2;   // Konstanten (einmal berechnet)
    T _beta1_t, _beta2_t;                   // beta^t, inkrementell fortgeschrieben
    T _bias_corr1, _bias_corr2;             // pro step() gesetzt, in update_parameter nur gelesen
};

#endif /* ADAM_HPP */
