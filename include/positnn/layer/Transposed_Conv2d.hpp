#ifndef TRANSPOSED_CONV2D_HPP
#define TRANSPOSED_CONV2D_HPP

// General headers
#include <cmath>
#include <iostream>

// Custom headers
#include "init.hpp"
#include "Layer.hpp"
#include "../tensor/transposed_convolution.hpp"
#include "../tensor/MixedTensor.hpp"
#include "../tensor/sum.hpp"
#include "../tensor/StdTensor.hpp"

template <typename OptimizerT, typename ForwardT=OptimizerT, typename BackwardT=ForwardT, typename GradientT=BackwardT>
class TransposedConv2d : public Layer<OptimizerT> {
public:
    TransposedConv2d(size_t _in_channels, size_t _out_channels, size_t _kernel_size, size_t _stride=1, size_t _padding=0, size_t _output_padding=0, size_t _dilation=1) :
        in_channels(_in_channels),
        out_channels(_out_channels),
        kernel_size(_kernel_size),
        stride(_stride),
        padding(_padding),
        output_padding(_output_padding),
        dilation(_dilation),
        // Gewichtsdimensionen bei TransposedConv: (in_channels, out_channels, kernel_h, kernel_w)
        weight({_in_channels, _out_channels, _kernel_size, _kernel_size}),
        bias(_out_channels),
        weight_gradient({_in_channels, _out_channels, _kernel_size, _kernel_size}),
        bias_gradient(_out_channels)
    {
        this->register_parameter(weight, weight_gradient);
        this->register_parameter(bias, bias_gradient);

        reset_parameters();
    }

    void reset_parameters() {
		// Wir nutzen float für die Berechnung der Bounds, um std::sqrt sicher zu verwenden
		// nonlinearity: leaky_relu mit a=0 ist identisch zu ReLU
		kaiming_uniform<ForwardT, float>(weight.get_forward(), 0, Mode::fan_in, NonLinearity::relu);

		// Initialisierung des Bias (PyTorch Style)
		if (!bias.get_forward().empty()) {
			float fan_in = calculate_correct_fan<ForwardT, float>(weight.get_forward(), Mode::fan_in);
			float bound = 1.0 / std::sqrt(fan_in);
			set_uniform<ForwardT, float>(bias.get_forward(), -bound, bound);
		}
		
		// Die Gewichte müssen auch für den Backward-Pass synchronisiert werden
		weight.get_backward() = weight.get_forward();

        std::cout << "Transposed_Conv2d Layer Kaiming initialized" << std::endl;
	}

    template <typename T>
    StdTensor<ForwardT> forward(StdTensor<T> const& x) {
        input = x;
        return transposed_convolution2d<ForwardT::nbits, ForwardT::es>(
            x, weight.get_forward(), bias.get_forward(), 
            stride, padding, output_padding, dilation
        );
    }

    template <typename T>
    StdTensor<BackwardT> backward(StdTensor<T> const& delta) {
        gradient(delta);
        return transposed_convolution2d_backward_input<BackwardT::nbits, BackwardT::es>(
            delta, weight.get_backward(), 
            input.shape()[2], input.shape()[3], 
            stride, stride, padding, padding
        );
    }

    // void gradient(StdTensor<GradientT> const& delta) {
    //     StdTensor<GradientT> temp_weight_gradient = transposed_convolution2d_gradient<GradientT::nbits, GradientT::es>(
    //         input, delta, stride, padding, dilation
    //     );
    //     StdTensor<GradientT> temp_bias_gradient = sum_last2(delta);

    //     // If there are many samples
    //     if(input.dim()>1 && input.shape()[0]>1){
    //         temp_weight_gradient /= input.shape()[0];

    //         temp_bias_gradient = sum_first(temp_bias_gradient);
    //         temp_bias_gradient /= input.shape()[0];
    //     }

    //     weight_gradient += temp_weight_gradient;
    //     bias_gradient += temp_bias_gradient;

    //     return;
    // }

    void gradient(StdTensor<GradientT> const& delta) {
        // 1. Berechnung der Roh-Gradienten (Summen über die räumlichen Dimensionen)
        StdTensor<GradientT> temp_weight_gradient = transposed_convolution2d_gradient<GradientT::nbits, GradientT::es>(
            input, delta, stride, padding, dilation
        );
        // sum_last2 summiert über H und W von delta -> Ergebnis: [Batch, Channels]
        StdTensor<GradientT> temp_bias_gradient = sum_last2(delta);

        // // 2. Räumliche Normalisierung berechnen
        // // Die räumlichen Dimensionen von delta entsprechen der Ausgabegröße des Layers
        // size_t out_h = delta.shape()[2];
        // size_t out_w = delta.shape()[3];
        // GradientT spatial_normalization = GradientT(out_h * out_w);

        // // Gradienten durch die Anzahl der Pixel teilen (Mean über die Fläche)
        // temp_weight_gradient /= spatial_normalization;
        // temp_bias_gradient /= spatial_normalization;

        // 3. Batch-Normalisierung (wie bisher)
        if(input.dim() > 1 && input.shape()[0] > 1){
            //GradientT batch_size = GradientT(input.shape()[0]);
            
            //temp_weight_gradient /= batch_size;

            // sum_first summiert über die Batch-Dimension -> Ergebnis: [Channels]
            temp_bias_gradient = sum_first(temp_bias_gradient);
            //temp_bias_gradient /= batch_size; //Keine Normalisierung
        }

        // 4. Akkumulation in die Layer-Parameter
        weight_gradient += temp_weight_gradient;
        bias_gradient += temp_bias_gradient;

        return;
    }

private:
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    size_t output_padding;
    size_t dilation;
    MixedTensor<OptimizerT, ForwardT, BackwardT> weight;
    MixedTensor<OptimizerT, ForwardT> bias;
    StdTensor<GradientT> input;
    StdTensor<OptimizerT> weight_gradient;
    StdTensor<OptimizerT> bias_gradient;
};

#endif /* TRANSPOSED_CONV2D_HPP */