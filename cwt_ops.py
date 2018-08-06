from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


def complex_morlet_wavelets(
        fb_array,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        flattening=False):
    """
    Computes the complex morlet wavelets

    This function computes the complex morlet wavelet defined as:
    PSI(k) = (pi*Fb)^(-0.5) * exp(i*2*pi*Fc*k) * exp(-(k^2)/Fb)
    It supports several values of Fb at once, while Fc is fixed to 1 since we can change the frequency of the
    wavelets by changing the scale. Note that greater Fb values will lead to more duration of the wavelet in time,
    leading to better frequency resolution but worse time resolution.
    Scales will be automatically computed from the given frequency range and the number of desired scales. The scales
    will increase exponentially.

    Args:
        fb_array: 1D array of values for Fb.
        fs: Sampling frequency of the input.
        lower_freq: Lower frequency to be considered for the scalogram.
        upper_freq: Upper frequency to be considered for the scalogram.
        n_scales: Number of scales to cover the frequency range
        flattening: (Optional) If True, each wavelet will be multiplied by its corresponding frequency, to avoid
         having too large coefficients for low frequency ranges, since it is common for natural signals to have a
         spectrum whose power decays roughly like 1/f. Defaults to False.

    Returns:
        wavelets: A list of computed wavelet banks.
        frequencies: Array of frequencies for each scale.
    """
    # Checking
    fb_array = np.array(fb_array)
    if fb_array.ndim != 1:
        raise Exception("Expected dimension for fb_array is 1")
    if lower_freq > upper_freq:
        raise Exception("lower_freq should be lower than upper_freq")
    if lower_freq < 0:
        raise Exception("Expected positive lower_freq.")

    n_scalograms = fb_array.shape[0]

    # Generate initial and last scale
    s_0 = fs / upper_freq
    s_n = fs / lower_freq

    # Generate the array of scales
    base = np.power(s_n / s_0, 1 / (n_scales - 1))
    scales = s_0 * np.power(base, np.arange(n_scales))

    # Generate the frequency range
    frequencies = fs / scales

    # Generate the wavelets
    wavelets = []
    for j in range(n_scalograms):
        fb = fb_array[j]
        one_side = int(scales[-1] * np.sqrt(5 * fb))
        kernel_size = 2 * one_side + 1
        wavelet_bank_real = np.zeros((1, kernel_size, 1, n_scales))
        wavelet_bank_imag = np.zeros((1, kernel_size, 1, n_scales))
        for i in range(n_scales):
            scale = scales[i]
            k_array = np.arange(kernel_size, dtype=np.float32) - one_side
            kernel_base = np.exp(-((k_array / scale) ** 2) / fb) / np.sqrt(np.pi * fb * scale)
            kernel_real = kernel_base * np.cos(2 * np.pi * k_array / scale)
            kernel_imag = kernel_base * np.sin(2 * np.pi * k_array / scale)
            if flattening:
                kernel_real = kernel_real * frequencies[i]
                kernel_imag = kernel_imag * frequencies[i]
            wavelet_bank_real[0, :, 0, i] = kernel_real
            wavelet_bank_imag[0, :, 0, i] = kernel_imag
        wavelets.append((wavelet_bank_real, wavelet_bank_imag))
    return wavelets, frequencies


def cwt_layer(
        inputs,
        wavelets,
        border_crop=0,
        stride=1,
        data_format="channel_last",
        name="cwt"):
    """
    CWT layer implementation in Tensorflow

    Implementation of CWT in Tensorflow, aimed at providing GPU acceleration. This layer use the computed wavelets.
    It supports the computation of several scalograms. Different scalograms will be stacked along the channel dimension.

    Args:
        inputs: A batch of 1D tensors of shape [batch_size, signal_size].
        wavelets: A list of computed wavelet banks.
        border_crop: (Optional) Int>=0 that specifies the number of samples to be removed at each border at the end.
         This parameter allows to input a longer signal than the final desired size to remove border effects of the CWT.
         Default is 0 (no removing at the borders).
        stride: (Optional) Int>0. The stride of the sliding window across the input. Default is 1.
        data_format: (Optional) A string from: "channel_last", "channel_first". Defaults to "channel_last".
         Specify the data format of the output data. With the default format "channel_last", the output has shape
         [batch, n_scales, signal_size, channels]. Alternatively, with the format "channel_first", the output has shape
         [batch, channels, n_scales, signal_size].
        name: (Optional) A name for the operation. Default is "cwt".

    Returns:
        Scalogram tensor.
    """
    # Checking
    if data_format != "channel_first" and data_format != "channel_last":
        raise Exception("Expected 'channel first' or 'channel_last' for data_format")
    n_scalograms = len(wavelets)

    # Generate the scalograms
    border_crop = int(border_crop/stride)
    start = border_crop
    if border_crop == 0:
        end = None
    else:
        end = -border_crop
    with tf.variable_scope(name):
        # Reshape input
        inputs_expand = tf.expand_dims(inputs, axis=1)
        inputs_expand = tf.expand_dims(inputs_expand, axis=3)
        scalograms_list = []
        for j in range(n_scalograms):
            with tf.name_scope(name + "_" + str(j)):
                bank_real, bank_imag = wavelets[j]
                out_real = tf.nn.conv2d(input=inputs_expand, filter=bank_real, strides=[1, 1, stride, 1],
                                        padding="SAME")
                out_imag = tf.nn.conv2d(input=inputs_expand, filter=bank_imag, strides=[1, 1, stride, 1],
                                        padding="SAME")
                out_real_crop = out_real[:, :, start:end, :]
                out_imag_crop = out_imag[:, :, start:end, :]
                out_power = tf.sqrt(tf.square(out_real_crop) + tf.square(out_imag_crop))
                single_scalogram = tf.transpose(out_power, perm=[0, 3, 2, 1])
                scalograms_list.append(single_scalogram)
        scalograms = tf.concat(scalograms_list, -1)
        if data_format == "channel_first":
            scalograms = tf.transpose(scalograms, perm=[0, 3, 1, 2])
    return scalograms
