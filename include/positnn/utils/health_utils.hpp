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

struct TensorStats {
    double max_val = -std::numeric_limits<double>::max();
    double min_val = std::numeric_limits<double>::max();
    double min_pos_nonzero = std::numeric_limits<double>::max();
    double max_neg_nonzero = -std::numeric_limits<double>::max();
    double mean = 0.0;
    double stddev = 0.0;
    size_t nar_count = 0;
    size_t zero_count = 0;
    size_t total_elements = 0;

    double zero_ratio() const {
        return total_elements > 0 ? (100.0 * zero_count / total_elements) : 0.0;
    }
};

template<typename T>
TensorStats inspect_tensor_detailed(const StdTensor<T>& tensor) {
    TensorStats stats;
    stats.total_elements = tensor.size();
    if (stats.total_elements == 0) return stats;

    double sum = 0.0;
    size_t valid_count = 0;

    for (size_t i = 0; i < stats.total_elements; ++i) {
        if (tensor[i].isnar()) {
            stats.nar_count++;
            continue;
        }

        double val = static_cast<double>(tensor[i]);
        if (val == 0.0) {
            stats.zero_count++;
        } else {
            if (val > 0.0 && val < stats.min_pos_nonzero) stats.min_pos_nonzero = val;
            if (val < 0.0 && val > stats.max_neg_nonzero) stats.max_neg_nonzero = val;
        }

        if (val > stats.max_val) stats.max_val = val;
        if (val < stats.min_val) stats.min_val = val;

        sum += val;
        valid_count++;
    }

    if (valid_count > 0) {
        stats.mean = sum / valid_count;
        double variance_sum = 0.0;
        for (size_t i = 0; i < stats.total_elements; ++i) {
            if (!tensor[i].isnar()) {
                double val = static_cast<double>(tensor[i]);
                variance_sum += (val - stats.mean) * (val - stats.mean);
            }
        }
        stats.stddev = std::sqrt(variance_sum / valid_count);
    }

    if (stats.min_pos_nonzero == std::numeric_limits<double>::max()) stats.min_pos_nonzero = 0.0;
    if (stats.max_neg_nonzero == -std::numeric_limits<double>::max()) stats.max_neg_nonzero = 0.0;
    if (stats.max_val == -std::numeric_limits<double>::max()) stats.max_val = 0.0;
    if (stats.min_val == std::numeric_limits<double>::max()) stats.min_val = 0.0;

    return stats;
}

template<typename T>
double calculate_update_ratio(const std::vector<double>& old_weights_raw, const StdTensor<T>& new_weights) {
    if (old_weights_raw.size() != new_weights.size() || old_weights_raw.empty()) return 0.0;
    size_t changed = 0;
    for (size_t i = 0; i < old_weights_raw.size(); ++i) {
        if (old_weights_raw[i] != static_cast<double>(new_weights[i])) {
            changed++;
        }
    }
    return (100.0 * changed) / old_weights_raw.size();
}

template<typename T>
void log_detailed_health(Layer<T>& net, const std::vector<double>& update_ratios = {}) {
    auto& all_params = net.parameters();
    
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "                             DETAILED POSIT HEALTH REPORT                             \n";
    std::cout << std::string(100, '=') << "\n";

    for (size_t i = 0; i < all_params.size(); ++i) {
        auto& p = all_params[i];
        TensorStats w_stats = inspect_tensor_detailed(p.weight);
        TensorStats g_stats = inspect_tensor_detailed(p.gradient);

        std::cout << "Layer " << i << " Parameters (W: " << p.weight.size() << " | G: " << p.gradient.size() << " elements):\n";
        
        // Scientific notation, so that underflow stays visible
        std::cout << std::scientific << std::setprecision(4);
        
        std::cout << "  [WEIGHTS]  "
                  << "Min: " << std::setw(12) << w_stats.min_val << " | "
                  << "Max: " << std::setw(12) << w_stats.max_val << " | "
                  << "Mean: " << std::setw(12) << w_stats.mean << " | "
                  << "StdDev: " << std::setw(12) << w_stats.stddev << "\n";
        std::cout << "             "
                  << "Closest to 0 (+/-): [+" << w_stats.min_pos_nonzero << ", " << w_stats.max_neg_nonzero << "]\n";
        
        // Back to plain formatting for the percentages
        std::cout << std::defaultfloat << std::fixed << std::setprecision(2);
        std::cout << "             "
                  << "Zeros: " << w_stats.zero_ratio() << "% | "
                  << "NaRs: " << w_stats.nar_count << "\n";

        // Scientific notation again for the very small gradients
        std::cout << std::scientific << std::setprecision(4);
        
        std::cout << "  [GRADS]    "
                  << "Min: " << std::setw(12) << g_stats.min_val << " | "
                  << "Max: " << std::setw(12) << g_stats.max_val << " | "
                  << "Mean: " << std::setw(12) << g_stats.mean << " | "
                  << "StdDev: " << std::setw(12) << g_stats.stddev << "\n";
        std::cout << "             "
                  << "Closest to 0 (+/-): [+" << g_stats.min_pos_nonzero << ", " << g_stats.max_neg_nonzero << "]\n";
        
        std::cout << std::defaultfloat << std::fixed << std::setprecision(2);
        std::cout << "             "
                  << "Zeros: " << g_stats.zero_ratio() << "% | "
                  << "NaRs: " << g_stats.nar_count << "\n";

        if (i < update_ratios.size()) {
            std::cout << "  [UPDATE]   " 
                      << "Ratio of weights modified in last step: " 
                      << update_ratios[i] << "%\n";
        }
        std::cout << std::string(100, '-') << "\n";
        
        // Reset for the next iteration
        std::cout << std::defaultfloat; 
    }
}

#endif // HEALTH_UTILS_HPP