# cmorlet-tensorflow

A TensorFlow implementation of the Continuous Wavelet Transform obtained via the complex Morlet wavelet. Please see the demo Jupyter Notebook for a demonstration on how it is used. This implementation is aimed to leverage GPU acceleration for the computation of the CWT in TensorFlow models. Please note that this implementation was made before TensorFlow 2, so you need any TensorFlow 1 version (i.e. tf 1.x).

This module was used to obtain the CWT of EEG signals for the RED-CWT model, described in:
"RED: Deep Recurrent Neural Networks for Sleep EEG Event Detection"
Nicolás I. Tapia, Pablo A. Estévez
2020 International Joint Conference on Neural Networks (IJCNN)
https://arxiv.org/abs/2005.07795
