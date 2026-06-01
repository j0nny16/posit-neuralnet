#ifndef CONV2D_HPP
#define CONV2D_HPP

// General headers
#include <cmath>
#include <vector>

// Custom headers
#include "init.hpp"
#include "Layer.hpp"
#include "../tensor/convolution.hpp"
#include "../tensor/MixedTensor.hpp"
#include "../tensor/sum.hpp"
#include "../tensor/StdTensor.hpp"

template <typename OptimizerT, typename ForwardT=OptimizerT, typename BackwardT=ForwardT, typename GradientT=BackwardT>
class Conv2d : public Layer<OptimizerT> {
// TODO: implement dilation
public:
	Conv2d(size_t _in_channels, size_t _out_channels, size_t _kernel_size, size_t _stride=1, size_t _padding=0, size_t _dilation=1) :
		in_channels(_in_channels),
		out_channels(_out_channels),
		kernel_size(_kernel_size),
		stride(_stride),
		padding(_padding),
		dilation(_dilation),
		weight({out_channels, in_channels, kernel_size, kernel_size}),
		bias(out_channels),
		weight_gradient({out_channels, in_channels, kernel_size, kernel_size}),
		bias_gradient(out_channels)
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

		std::cout << "Conv2d Layer Kaiming initialized" << std::endl;
	}

	template <typename T>
	StdTensor<ForwardT> forward(StdTensor<T> const& x) {
		input = x;

		saved_input_shape = input.shape(); // Input Shape merken
		// std::cerr << "Forward Called! Saved Input Shape: [";
		// for (auto s : saved_input_shape)
		// {
		// 	std::cerr << s << ", ";
		// }
		// std::cerr << "]" << std::endl;
		return convolution2d<ForwardT::nbits, ForwardT::es>(x, weight.get_forward(), bias.get_forward(), stride, padding, 1, dilation, &w1);
	}

	// template <typename T>
	// StdTensor<BackwardT> backward(StdTensor<T> const& delta) {
	// 	gradient(delta);
	// 	StdTensor<BackwardT> rotated = rotate_weight(weight.get_backward());
	// 	return convolution2d<BackwardT::nbits, BackwardT::es>(delta, rotated, StdTensor<BackwardT>(), 1, (kernel_size-1)*dilation-padding, stride, dilation, &w3);
	// }

	template <typename T>
	StdTensor<BackwardT> backward(StdTensor<T> const& delta) {
		// 1. Berechnet die Gradienten für Gewichte und Bias
		gradient(delta);
		
		// 2. Berechnet den Gradienten bezüglich des Inputs (Das ist der Wert, der evtl. zu klein ist)
		StdTensor<BackwardT> rotated = rotate_weight(weight.get_backward());
		StdTensor<BackwardT> input_gradient = convolution2d<BackwardT::nbits, BackwardT::es>(
			delta, rotated, StdTensor<BackwardT>(), 1, (kernel_size-1)*dilation-padding, stride, dilation, &w3
		);

		// 3. DIMENSION FIX: Gradienten auffüllen, falls Stride > 1 Pixel geschluckt hat
		if (saved_input_shape.empty()) {
        	throw std::runtime_error("[Conv2d Fatal Error] saved_input_shape is empty!");
    }
		
		if (input_gradient.shape() != this->saved_input_shape) {
			
			// Leeren Tensor in der exakten Zielgröße erstellen
			StdTensor<BackwardT> corrected_grad(this->saved_input_shape);
			
			// Zur Sicherheit alles explizit mit 0 initialisieren
			corrected_grad.set(BackwardT(0));
			
			// Grenzen des berechneten (zu kleinen) Gradienten
			size_t b_max = input_gradient.shape()[0];
			size_t c_max = input_gradient.shape()[1];
			size_t h_max = input_gradient.shape()[2];
			size_t w_max = input_gradient.shape()[3];

			// Die vorhandenen Werte rüberkopieren. 
			// Die fehlenden Zeilen/Spalten am rechten und unteren Rand bleiben 0.
			for(size_t b = 0; b < b_max; ++b) {
				for(size_t c = 0; c < c_max; ++c) {
					for(size_t h = 0; h < h_max; ++h) {
						for(size_t w = 0; w < w_max; ++w) {
							corrected_grad[{b, c, h, w}] = input_gradient[{b, c, h, w}];
						}
					}
				}
			}
			
			// Den zu kleinen Tensor mit der korrigierten Version überschreiben
			input_gradient = corrected_grad;
		}

		return input_gradient;
	}

	void gradient(StdTensor<GradientT> const& delta) {
		StdTensor<GradientT> temp_weight_gradient = convolution2d_gradient(input, delta, stride, padding, dilation, &w2);
		StdTensor<GradientT> temp_bias_gradient = sum_last2(delta);

		// If there are many samples
		if(input.dim()>1 && input.shape()[0]>1){
			temp_bias_gradient = sum_first(temp_bias_gradient);
			// Divisionen im Gradient werden ausgeschalten, das übernimmt der Loss
			// temp_weight_gradient /= input.shape()[0];

			// temp_bias_gradient = sum_first(temp_bias_gradient);
			// temp_bias_gradient /= input.shape()[0];
		}

		weight_gradient += temp_weight_gradient;
		bias_gradient += temp_bias_gradient;

		return;
	}

	// void gradient(StdTensor<GradientT> const& delta) {
	// 	// 1. Berechnung der Roh-Gradienten für Gewichte und Bias
	// 	// convolution2d_gradient berechnet die Korrelation zwischen Input und Delta
	// 	StdTensor<GradientT> temp_weight_gradient = convolution2d_gradient<GradientT::nbits, GradientT::es>(
	// 		input, delta, stride, padding, dilation
	// 	);
	// 	// sum_last2 summiert delta über die räumlichen Dimensionen (H_out, W_out) -> Ergebnis: [Batch, Channels]
	// 	StdTensor<GradientT> temp_bias_gradient = sum_last2(delta);

	// 	// 2. Räumliche Normalisierung (Spatial Mean)
	// 	// Die räumlichen Dimensionen von delta entsprechen der Ausgabegröße des Layers (H_out * W_out)
	// 	size_t out_h = delta.shape()[2];
	// 	size_t out_w = delta.shape()[3];
	// 	GradientT spatial_normalization = GradientT(out_h * out_w);

	// 	// Division durch die Anzahl der Pixel, um den mittleren Fehler pro Pixel zu erhalten
	// 	temp_weight_gradient /= spatial_normalization;
	// 	temp_bias_gradient /= spatial_normalization;

	// 	// 3. Batch-Normalisierung (Batch Mean)
	// 	if(input.dim() > 1 && input.shape()[0] > 1){
	// 		GradientT batch_size = GradientT(input.shape()[0]);
			
	// 		// Gewichts-Gradienten durch Batch-Größe teilen
	// 		temp_weight_gradient /= batch_size;

	// 		// Bias-Gradienten über die Batch-Dimension aufsummieren und dann teilen
	// 		temp_bias_gradient = sum_first(temp_bias_gradient);
	// 		temp_bias_gradient /= batch_size;
	// 	}

	// 	// 4. Akkumulation in die globalen Gradienten-Tensor des Layers
	// 	weight_gradient += temp_weight_gradient;
	// 	bias_gradient += temp_bias_gradient;

	// 	return;
	// }

private:
	size_t in_channels;
	size_t out_channels;
	size_t kernel_size;
	size_t stride;
	size_t padding;
	size_t dilation;
	MixedTensor<OptimizerT, ForwardT, BackwardT> weight;
	MixedTensor<OptimizerT, ForwardT> bias;
	StdTensor<GradientT> input;
	StdTensor<OptimizerT> weight_gradient;
	StdTensor<OptimizerT> bias_gradient;
	Window w1, w2, w3;
	std::vector<size_t> saved_input_shape; // Input Shape merken
};

#endif /* CONV2D_HPP */
