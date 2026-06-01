#ifndef ADAM_HPP
#define ADAM_HPP

//#include <vector>
//#include <cmath>
#include "../layer/Parameter.hpp"
#include "../optimizer/Optimizer.hpp"
#include "../tensor/StdTensor.hpp"

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
        _options(options0)
    {
        /* * INITIALISIERUNG:
         * Anders als SGD benötigt Adam zwei Status-Buffer (m und v) pro Parameter.
         * Da StdTensor beim einfachen resize() keinen Speicher für Elemente reserviert,
         * müssen wir hier explizit die Dimensionen des Ziel-Parameters übergeben.
         */
        for (auto const& p : this->_parameters) {
            _m.push_back(StdTensor<T>(p.weight.shape()));
            _v.push_back(StdTensor<T>(p.weight.shape()));
            
            _m.back().clear(); 
            _v.back().clear();
        }
    }

    void step() {
        /*
         * GLOBALER ZEITSCHRITT:
         * t muss einmal pro Batch erhöht werden, nicht pro Parameter-Update.
         * Danach rufen wir die Basisklasse auf, die das Multithreading startet.
         */
        _options.t++;
        Optimizer<T>::step(); 
    }

    AdamOptions<T>& options() { return _options; }

private:
    void update_parameter(Parameter<T>& p, size_t const i) override {
        /*
         * MULTITHREADING-KONTEXT:
         * Diese Funktion wird von einem Thread für EINEN spezifischen Parameter (Layer) aufgerufen.
         * Das Objekt 'p' enthält alle Gewichte dieses Layers (z.B. eine 784x512 Matrix).
         */
        StdTensor<T>& weights = p.weight;
        StdTensor<T>& grad = p.gradient;
        StdTensor<T>& m = _m[i];
        StdTensor<T>& v = _v[i];

        T beta1 = _options.beta_1;
        T beta2 = _options.beta_2;
        T eps = _options.eps;
        T lr = _options.learning_rate;
        T const one = T(1.0);

        // Vorberechnen der Bias-Korrektur (Skalare)
        auto posit_pow = [](T base, size_t exp) {
            T res = T(1.0);
            for (size_t k = 0; k < exp; ++k) res *= base;
            return res;
        };

        T bias_corr1 = one / (one - posit_pow(beta1, _options.t));
        T bias_corr2 = one / (one - posit_pow(beta2, _options.t));

        /*
         * WARUM EIN EXPLIZITER LOOP STATT FUSED()?
         * * 1. Funktionalität: Die Methode fused() in PositNN kann nur (A = A*x + B).
         * Adam benötigt aber Wurzeln (sqrt) und Divisionen. Diese Operationen
         * existieren nicht als 'fused'-Tensor-Befehle in StdTensor.hpp.
         * * 2. Cache-Effizienz (Kernel Fusion): 
         * Würden wir mehrere separate Tensor-Befehle nutzen (einen für m, einen für v, 
         * einen für sqrt, etc.), müsste die CPU die riesigen Datenmengen 5-6 Mal aus 
         * dem RAM lesen und zurückschreiben (Memory Bandwidth Bottleneck).
         * * 3. Der manuelle Loop erlaubt es, ein einzelnes Gewicht j in ein CPU-Register zu laden,
         * ALLE Adam-Berechnungen durchzuführen und das Ergebnis EINMAL zurückzuschreiben.
         * Das ist um ein Vielfaches schneller.
         */
        for (size_t j = 0; j < weights.size(); ++j) {
            T g = grad[j];

            // Update m (1. Moment - Momentum)
            m[j] = beta1 * m[j] + (T(1.0) - beta1) * g;

            // Update v (2. Moment - Unzentrierte Varianz)
            v[j] = beta2 * v[j] + (T(1.0) - beta2) * (g * g);

            // Bias-Korrektur anwenden
            T m_hat = m[j] * bias_corr1;
            T v_hat = v[j] * bias_corr2;

            // Das eigentliche Gewichtsupdate
            using sw::unum::sqrt;

            T denom = sqrt(v_hat) + eps;

            // DEBUG: Prüfung auf Division durch Null oder NaR
            if (denom == T(0) || weights[j].isnar() || denom.isnar()) {
                std::cerr << "0 or NaR" << std::endl;
                continue; 
            }

            //weights[j] = weights[j] - (lr * m_hat) / (sqrt(v_hat) + eps);
            weights[j] = weights[j] - (lr * m_hat) / denom;
        }

        p.update();
    }

    AdamOptions<T> _options;
    std::vector<StdTensor<T>> _m; 
    std::vector<StdTensor<T>> _v; 
};

#endif