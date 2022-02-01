# Sharpened Cosine Distance

PyTorch implementation of the Sharpened Cosine Distance operator.

The core idea came from Brandon Rohrer([@_brohrer_](https://twitter.com/_brohrer_)) and the implementation 
is based on the tf/keras implementation of [Raphael Pisoni](https://www.rpisoni.dev/posts/cossim-convolution/).

This implementation supports

- 2D operation only
- asymmetric kernels, any shape
- CUDA / GPU


If you find this implementation useful please give it a star. Open issues for bugs/ideas.

If you are planning to build something on top of it let me know, I am always up for some good collaborations ;) 