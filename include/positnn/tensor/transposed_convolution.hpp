#ifndef TRANSPOSED_CONVOLUTION_HPP
#define TRANSPOSED_CONVOLUTION_HPP

#ifdef LL_THREADS
	#if LL_THREADS>1
		#define USING_LL_THREADS
	#endif
#endif /* LL_THREADS */

// General headers
#ifdef USING_LL_THREADS
#include <thread>
#endif /* USING_LL_THREADS */
#include <universal/posit/posit>
#include <vector>

// Custom headers
#include "StdTensor.hpp"
#include "Window.hpp"
#include "../utils/Quire.hpp"

// Namespaces
using namespace sw::unum;

template <size_t nbits, size_t es>
// gather anstatt push operation wegen quire
inline void do_transposed_convolution(StdTensor<posit<nbits, es>> const& input,
                                      StdTensor<posit<nbits, es>> const& kernel,
                                      Quire<nbits, es>& q,
                                      size_t const input_batch_offset,
                                      size_t const kernel_out_channel_offset,
                                      size_t const in_channels,
                                      size_t const in_h, size_t const in_w,
                                      size_t const kernel_h, size_t const kernel_w,
                                      size_t const stride, size_t const padding, size_t const dilation,
                                      size_t const oy, size_t const ox) {
    
    // Strides for tensor access
    size_t const input_channel_stride = input.strides()[1];
    size_t const input_h_stride = input.strides()[2];
    size_t const kernel_in_channel_stride = kernel.strides()[0];
    size_t const kernel_h_stride = kernel.strides()[2];

    // Gather loop over kernel dimensions
    for (size_t ky = 0; ky < kernel_h; ++ky) {
        ptrdiff_t const y_unstrided = static_cast<ptrdiff_t>(oy + padding) - static_cast<ptrdiff_t>(ky * dilation);
        if (y_unstrided < 0 || y_unstrided % static_cast<ptrdiff_t>(stride) != 0) continue;
        
        ptrdiff_t const iy = y_unstrided / static_cast<ptrdiff_t>(stride);

        if (iy < 0 || static_cast<size_t>(iy) >= in_h) continue;

        for (size_t kx = 0; kx < kernel_w; ++kx) {
            ptrdiff_t const x_unstrided = static_cast<ptrdiff_t>(ox + padding) - static_cast<ptrdiff_t>(kx * dilation);
            if (x_unstrided < 0 || x_unstrided % static_cast<ptrdiff_t>(stride) != 0) continue;
            
            size_t const ix = static_cast<size_t>(x_unstrided) / stride; //TODO: cast wie oben anpassen
            if (ix >= in_w) continue;

            size_t input_channel_idx = input_batch_offset + (iy * input_h_stride) + ix;
            size_t kernel_in_channel_idx = kernel_out_channel_offset + (ky * kernel_h_stride) + kx;

            for (size_t channel = 0; channel < in_channels; ++channel) {
                q += Quire_mul(input[input_channel_idx], kernel[kernel_in_channel_idx]);
                input_channel_idx += input_channel_stride;
                kernel_in_channel_idx += kernel_in_channel_stride;
            }
        }
    }
}



template <size_t nbits, size_t es>
void transposed_convolution2d_thread(StdTensor<posit<nbits, es>> const& input,
                                     StdTensor<posit<nbits, es>> const& weight,
                                     StdTensor<posit<nbits, es>> const& bias,
                                     StdTensor<posit<nbits, es>>& output,
                                     size_t const stride, size_t const padding, size_t const dilation,
                                     size_t input_batch_idx, size_t output_batch_idx,
                                     size_t const n_samples) {
    
    bool const no_bias = bias.empty();
    size_t const in_channels = weight.shape()[0];
    size_t const out_channels = weight.shape()[1];
    size_t const kernel_h = weight.shape()[2];
    size_t const kernel_w = weight.shape()[3];
    
    size_t const in_h = input.shape()[2];
    size_t const in_w = input.shape()[3];
    size_t const out_h = output.shape()[2];
    size_t const out_w = output.shape()[3];

    size_t const input_batch_stride = input.strides()[0];
    size_t const output_batch_stride = output.strides()[0];
    size_t const output_channel_stride = output.strides()[1];
    size_t const weight_out_channel_stride = weight.strides()[1];

    Quire<nbits, es> q;

    for (size_t i = 0; i < n_samples; ++i) {
        size_t output_channel_idx = output_batch_idx;
        size_t weight_out_channel_idx = 0;

        for (size_t oc = 0; oc < out_channels; ++oc) {
            size_t out_spatial_idx = 0;

            for (size_t oy = 0; oy < out_h; ++oy) {
                for (size_t ox = 0; ox < out_w; ++ox) {
                    
                    if (no_bias) q.clear();
                    else q = bias[oc];

                    do_transposed_convolution(input, weight, q, 
                                                     input_batch_idx, weight_out_channel_idx, 
                                                     in_channels, in_h, in_w, kernel_h, kernel_w, 
                                                     stride, padding, dilation, oy, ox);

                    convert(q.to_value(), output[output_channel_idx + out_spatial_idx]);
                    out_spatial_idx++;
                }
            }
            output_channel_idx += output_channel_stride;
            weight_out_channel_idx += weight_out_channel_stride;
        }
        input_batch_idx += input_batch_stride;
        output_batch_idx += output_batch_stride;
    }
}

template <size_t nbits, size_t es>
StdTensor<posit<nbits, es>> transposed_convolution2d(StdTensor<posit<nbits, es>> const& input,
                                                     StdTensor<posit<nbits, es>> const& weight,
                                                     StdTensor<posit<nbits, es>> const& bias,
                                                     size_t const stride=1,
                                                     size_t const padding=0,
                                                     size_t const output_padding=0,
                                                     size_t const dilation=1) {

    size_t const batch_size = input.shape()[0];
    //size_t const in_channels = weight.shape()[0];
    size_t const out_channels = weight.shape()[1];
    size_t const kernel_h = weight.shape()[2];
    size_t const kernel_w = weight.shape()[3];

    size_t const in_h = input.shape()[2];
    size_t const in_w = input.shape()[3];

    size_t const out_h = (in_h - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + 1 + output_padding;
    size_t const out_w = (in_w - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + 1 + output_padding;

    StdTensor<posit<nbits, es>> output({batch_size, out_channels, out_h, out_w});

    size_t const input_batch_stride = input.strides()[0];
    size_t const output_batch_stride = output.strides()[0];

#ifdef USING_LL_THREADS

    const size_t max_threads = (LL_THREADS < batch_size) ? LL_THREADS : batch_size;
    std::vector<std::thread> threads;
    threads.reserve(max_threads);

    size_t const n_samples = batch_size / max_threads;
    size_t const nthreads_more = batch_size % max_threads;

    size_t input_samples_begin = 0;
    size_t output_samples_begin = 0;

    for(size_t t = 0; t < max_threads; ++t){
        size_t thread_samples = n_samples + (t < nthreads_more ? 1 : 0);

        threads.push_back(std::thread(transposed_convolution2d_thread<nbits, es>,
                                      std::cref(input), std::cref(weight), std::cref(bias), std::ref(output),
                                      stride, padding, dilation,
                                      input_samples_begin, output_samples_begin, thread_samples));
        
        input_samples_begin += thread_samples * input_batch_stride;
        output_samples_begin += thread_samples * output_batch_stride;
    }
    
    for(std::thread& t : threads) {
        t.join();
    }

#else

    transposed_convolution2d_thread<nbits, es>(std::cref(input), std::cref(weight), std::cref(bias), std::ref(output),
                                      stride, padding, dilation,
                                      0, 0, batch_size);

#endif

    return output;
}

template <size_t nbits, size_t es>
void transposed_convolution2d_backward_input_thread(
    StdTensor<posit<nbits, es>> const& grad_output,
    StdTensor<posit<nbits, es>> const& weight,
    StdTensor<posit<nbits, es>>& grad_input,
    size_t const stride_h, size_t const stride_w,
    size_t const pad_h, size_t const pad_w,
    size_t const dilation,
    size_t batch_begin, size_t const n_samples) {

    size_t const in_channels = grad_input.shape()[1];
    size_t const in_height = grad_input.shape()[2];
    size_t const in_width = grad_input.shape()[3];
    
    size_t const out_channels = grad_output.shape()[1];
    size_t const out_height = grad_output.shape()[2];
    size_t const out_width = grad_output.shape()[3];
    
    size_t const kernel_h = weight.shape()[2];
    size_t const kernel_w = weight.shape()[3];

    size_t const grad_out_batch_stride = grad_output.strides()[0];
    size_t const grad_out_channel_stride = grad_output.strides()[1];
    size_t const grad_in_batch_stride = grad_input.strides()[0];
    size_t const grad_in_channel_stride = grad_input.strides()[1];
    size_t const weight_in_channel_stride = weight.strides()[0]; 
    size_t const weight_out_channel_stride = weight.strides()[1];

    Quire<nbits, es> q;

    size_t input_batch = batch_begin * grad_in_batch_stride;
    size_t output_batch = batch_begin * grad_out_batch_stride;

    for (size_t b = 0; b < n_samples; b++) {
        size_t input_channel = input_batch;
        size_t weight_in_channel = 0;

        for (size_t ic = 0; ic < in_channels; ic++) {
            size_t spatial_idx = 0;
            
            for (size_t ih = 0; ih < in_height; ih++) {
                for (size_t iw = 0; iw < in_width; iw++) {
                    q.clear();
                    
                    size_t output_channel = output_batch;
                    size_t weight_out_channel = weight_in_channel;

                    for (size_t oc = 0; oc < out_channels; oc++) {
                        size_t w_idx = weight_out_channel;
                        
                        for (size_t kh = 0; kh < kernel_h; kh++) {
                            for (size_t kw = 0; kw < kernel_w; kw++) {
                                int oh = static_cast<int>(ih * stride_h + kh * dilation) - static_cast<int>(pad_h);
                                int ow = static_cast<int>(iw * stride_w + kw * dilation) - static_cast<int>(pad_w);

                                if (oh >= 0 && static_cast<size_t>(oh) < out_height && ow >= 0 && static_cast<size_t>(ow) < out_width) {
                                    size_t grad_out_idx = output_channel + oh * out_width + ow;
                                    q += Quire_mul(grad_output[grad_out_idx], weight[w_idx]);
                                }
                                w_idx++;
                            }
                        }
                        output_channel += grad_out_channel_stride;
                        weight_out_channel += weight_out_channel_stride;
                    }
                    
                    convert(q.to_value(), grad_input[input_channel + spatial_idx]);
                    spatial_idx++;
                }
            }
            input_channel += grad_in_channel_stride;
            weight_in_channel += weight_in_channel_stride;
        }
        input_batch += grad_in_batch_stride;
        output_batch += grad_out_batch_stride;
    }
}

template <size_t nbits, size_t es>
StdTensor<posit<nbits, es>> transposed_convolution2d_backward_input(
    StdTensor<posit<nbits, es>> const& grad_output,
    StdTensor<posit<nbits, es>> const& weight,
    size_t const in_height, size_t const in_width,
    size_t const stride_h = 1, size_t const stride_w = 1,
    size_t const pad_h = 0, size_t const pad_w = 0,
    size_t const dilation = 1
    ) {

    size_t const batch_size = grad_output.shape()[0];
    size_t const in_channels = weight.shape()[0]; 

    StdTensor<posit<nbits, es>> grad_input({batch_size, in_channels, in_height, in_width});

#ifdef USING_LL_THREADS
    const size_t max_threads = (LL_THREADS < batch_size) ? LL_THREADS : batch_size;
    std::vector<std::thread> threads;
    threads.reserve(max_threads);

    size_t const n_samples = batch_size / max_threads;
    size_t const nthreads_more = batch_size % max_threads;

    size_t batch_begin = 0;

    for (size_t t = 0; t < max_threads; t++) {
        size_t thread_samples = n_samples;
        if (t < nthreads_more)
            thread_samples++;

        threads.push_back(std::thread(transposed_convolution2d_backward_input_thread<nbits, es>,
                                      std::cref(grad_output), std::cref(weight), std::ref(grad_input),
                                      stride_h, stride_w, pad_h, pad_w, dilation,
                                      batch_begin, thread_samples));

        batch_begin += thread_samples;
    }

    for (std::thread& t : threads) {
        t.join();
    }

#else
    transposed_convolution2d_backward_input_thread<nbits, es>(std::cref(grad_output), std::cref(weight), std::ref(grad_input),
                                      stride_h, stride_w, pad_h, pad_w, dilation,
                                      0, batch_size);
#endif

    return grad_input;
}


template <size_t nbits, size_t es>
void transposed_convolution2d_gradient_thread( StdTensor<posit<nbits, es>> const& input,
                                               StdTensor<posit<nbits, es>> const& delta,
                                               StdTensor<posit<nbits, es>>& dweight,
                                               size_t const stride,
                                               size_t const padding,
                                               size_t const dilation,
                                               size_t const dweight_begin, size_t const n_samples ) {

    size_t const batch_size = input.shape()[0];
    size_t const in_channels = input.shape()[1];
    size_t const in_h = input.shape()[2];
    size_t const in_w = input.shape()[3];

    size_t const out_channels = delta.shape()[1];
    size_t const out_h = delta.shape()[2];
    size_t const out_w = delta.shape()[3];

    //size_t const kernel_h = dweight.shape()[2];
    size_t const kernel_w = dweight.shape()[3];
    size_t const size = dweight.strides()[1]; // kernel_h * kernel_w

    // Initial indices für dweight Shape: [in_channels, out_channels, kernel_h, kernel_w]
    size_t in_channel0 = dweight_begin / dweight.strides()[0];
    size_t out_channel0 = (dweight_begin % dweight.strides()[0]) / dweight.strides()[1];
    size_t idx0 = dweight_begin % dweight.strides()[1];

    Quire<nbits, es> q;
    size_t n = dweight_begin;
    size_t const n_end = dweight_begin + n_samples;
    bool first = true;

    for (size_t in_c = in_channel0; in_c < in_channels; in_c++) {
        for (size_t out_c = (first) ? out_channel0 : 0; out_c < out_channels; out_c++) {
            for (size_t idx = (first) ? idx0 : 0; idx < size; idx++) {
                
                size_t kh = idx / kernel_w;
                size_t kw = idx % kernel_w;

                q.clear();

                for (size_t b = 0; b < batch_size; ++b) {
                    for (size_t ih = 0; ih < in_h; ++ih) {
                        for (size_t iw = 0; iw < in_w; ++iw) {
                            int oh = static_cast<int>(ih * stride) - static_cast<int>(padding) + static_cast<int>(kh * dilation);
                            int ow = static_cast<int>(iw * stride) - static_cast<int>(padding) + static_cast<int>(kw * dilation);

                            if (oh >= 0 && static_cast<size_t>(oh) < out_h && ow >= 0 && static_cast<size_t>(ow) < out_w) {
                                q += Quire_mul(input[{b, in_c, ih, iw}], delta[{b, out_c, static_cast<size_t>(oh), static_cast<size_t>(ow)}]);
                            }
                        }
                    }
                }

                convert(q.to_value(), dweight[n++]);

                if (n == n_end)
                    return;

                if (first)
                    first = false;
            }
        }
    }
}

template <size_t nbits, size_t es>
StdTensor<posit<nbits, es>> transposed_convolution2d_gradient(
                                                StdTensor<posit<nbits, es>> const& input,
                                                StdTensor<posit<nbits, es>> const& delta,
                                                size_t const stride=1,
                                                size_t const padding=0,
                                                size_t const dilation=1,
                                                size_t const kernel_h=0, size_t const kernel_w=0
                                             ) {

    size_t const in_channels = input.shape()[1];
    size_t const out_channels = delta.shape()[1];

    StdTensor<posit<nbits, es>> dweight({in_channels, out_channels, kernel_h, kernel_w});
    size_t const size_total = dweight.size();

#ifdef USING_LL_THREADS

    const size_t max_threads = (LL_THREADS < size_total) ? LL_THREADS : size_total;
    std::vector<std::thread> threads;
    threads.reserve(max_threads);

    size_t const n_samples = size_total / max_threads;
    size_t const nthreads_more = size_total % max_threads;
    size_t dweight_begin = 0;

    for (size_t t = 0; t < max_threads; t++) {
        size_t thread_samples = n_samples;
        if (t < nthreads_more)
            thread_samples++;

        threads.push_back(std::thread(transposed_convolution2d_gradient_thread<nbits, es>,
                                      std::cref(input), std::cref(delta), std::ref(dweight),
                                      stride, padding, dilation, dweight_begin, thread_samples));
        
        dweight_begin += thread_samples;
    }

    for (std::thread& t : threads) {
        t.join();
    }

#else

    transposed_convolution2d_gradient_thread<nbits, es>(std::cref(input), std::cref(delta), std::ref(dweight),
                                        stride, padding, dilation, 0, size_total);

#endif

    return dweight;
}

#endif /* TRANSPOSED_CONVOLUTION_HPP */
