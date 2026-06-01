#ifndef HEALTH_UTILS_HPP
#define HEALTH_UTILS_HPP

#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <limits>
#include <vector>

#include "../tensor/StdTensor.hpp"
#include "../layer/Layer.hpp"

// Stats and Overflow Detection

// Struktur zur Speicherung aller numerischen Kennzahlen eines Tensors
struct TensorStats {
    double max_abs = 0.0;
    double min_abs_nonzero = std::numeric_limits<double>::max();
    double mean = 0.0;
    size_t nar_count = 0;
    size_t zero_count = 0;
    size_t total_elements = 0;

    double zero_ratio() const {
        return total_elements > 0 ? (100.0 * zero_count / total_elements) : 0.0;
    }
};

template<typename T>
TensorStats inspect_tensor(const StdTensor<T>& tensor) {
    TensorStats stats;
    stats.total_elements = tensor.size();
    if (stats.total_elements == 0) return stats;

    double sum = 0.0;
    size_t valid_count = 0;

    for (size_t i = 0; i < tensor.size(); ++i) {
        // NaR-Erkennung (Priorität bei Posits)
        if (tensor[i].isnar()) {
            stats.nar_count++;
            continue;
        }

        double val = static_cast<double>(tensor[i]);
        double abs_val = std::abs(val);

        // Null- vs. Kleinstwert-Erkennung (Underflow-Metrik)
        if (abs_val == 0.0) {
            stats.zero_count++;
        } else {
            if (abs_val < stats.min_abs_nonzero) {
                stats.min_abs_nonzero = abs_val;
            }
        }

        // Maximalwert-Erkennung (Overflow-Metrik)
        if (abs_val > stats.max_abs) {
            stats.max_abs = abs_val;
        }

        sum += val;
        valid_count++;
    }

    // Mittelwert berechnen (ohne NaRs)
    if (valid_count > 0) {
        stats.mean = sum / valid_count;
    }
    
    // Falls keine Werte ungleich Null gefunden wurden, korrigieren
    if (stats.min_abs_nonzero == std::numeric_limits<double>::max()) {
        stats.min_abs_nonzero = 0.0;
    }

    return stats;
}

template<typename T>
bool validate_tensor(const StdTensor<T>& tensor, const std::string& label, const std::string& context) {
    
    #ifdef VALIDATE
    if(!VALIDATE) return true;
    #endif

    size_t first_nar_idx = 0;
    size_t total_nars = 0;
    bool found_nar = false;

    // Schneller Loop zur Lokalisierung
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (tensor[i].isnar()) {
            if (!found_nar) {
                first_nar_idx = i;
                found_nar = true;
            }
            total_nars++;
        }
    }

    if (found_nar) {
        std::cerr << "\n" << std::string(30, '!') << " NaR DETECTED " << std::string(30, '!') << std::endl;
        std::cerr << "CONTEXT: " << context << " | TENSOR: " << label << std::endl;
        std::cerr << "SHAPE:   ";
        for (auto s : tensor.shape()) std::cerr << s << " ";
        std::cerr << "\nFirst NaR at flat index: " << first_nar_idx << std::endl;
        std::cerr << "TOTAL NaRs in this tensor: " << total_nars << " / " << tensor.size() << std::endl;
        std::cerr << std::string(72, '!') << std::endl;
        return false; // Validierung fehlgeschlagen
    }
    
    return true; // Alles sauber
}

#endif //HEALTH_UTILS_HPP