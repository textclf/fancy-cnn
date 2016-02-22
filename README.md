# `fancy-cnn`
Multiparadigm Sequential Convolutional Neural Networks for text classification.

##General Considerations
First off, this class of sequential convolutional networks is quite `GEMM` intensive, and *really* isn't suited to a CPU. You also *really* should use `CuDNN` when training. 

Bleeding edge installations of Theano and Keras are required. The TensorFlow backend *has not* been tested with the fancier `TimeDistributed` architectures. When considering your hardware, note that...

* On a 2.6 GHz Intel Core i7, one epoch of IMDB training takes ~3.5 days
* On a GRID K520 without CuDNN, one epoch of IMDB training takes ~1 hour
* On a GRID K520 *with* CuDNN, one epoch of IMDB training takes ~30 minutes
* On a GTX Titan X with CuDNN, one epoch of IMDB training takes ~11 minutes

If you want to use `CuDNN`, you really should also put

```
[dnn]
conv.algo_fwd = time_on_shape_change
conv.algo_bwd = time_on_shape_change
```
to your `.theanorc`. This will let Theano (at compilation time) pick the fastest convolution implementation given your input and kernel size.

## IMDB Dataset

Look at (and run) `prepare_imdb*.py` to prepare your data! Look at [this](https://github.com/lukedeo/fancy-cnn/blob/master/examples/imdb) to see how to train the IMDB model.

## Yelp Humor Dataset

Look at (and run) `prepare_yelp*.py` to prepare your data! Look at [this](https://github.com/lukedeo/fancy-cnn/blob/master/examples/yelp) to see how to train a model on Yelp.

