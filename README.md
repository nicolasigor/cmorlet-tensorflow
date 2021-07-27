# cmorlet-tensorflow

A TensorFlow implementation of the Continuous Wavelet Transform obtained via the complex Morlet wavelet. Please see the demo Jupyter Notebook for usage demonstration and more implementation details. 

This implementation is aimed to leverage GPU acceleration for the computation of the CWT in TensorFlow models. The morlet's wavelet width can be set as a trainable parameter if you want to adjust it via backprop. This implementation now supports  TensorFlow 2.

This module was used to obtain the CWT of EEG signals for the RED-CWT model, described in:

N. I. Tapia and P. A. Estévez, "RED: Deep Recurrent Neural Networks for Sleep EEG Event Detection," *2020 International Joint Conference on Neural Networks (IJCNN)*, Glasgow, United Kingdom, 2020, pp. 1-8, doi: 10.1109/IJCNN48605.2020.9207719.

Full text available at: https://arxiv.org/abs/2005.07795

If you find this software useful, please consider citing our work.
