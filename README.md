# `fancy-cnn`
Convolutional Neural Networks for multi-sentence sentiment analysis (Stanford CS224N)

##General Considerations
First off, this class of sequential convolutional networks is quite `GEMM` intensive, and *really* isn't suited to a CPU. You also *really* should use `CuDNN` when training. 

Bleeding edge installations of Theano and Keras are required.

For reference: 

* On a 2.6 GHz Intel Core i7, one epoch of IMDB training takes >10,000 secocnds
* On a GRID K520 without CuDNN, one epoch of IMDB training takes ~2,000 secocnds
* On a GRID K520 *with* CuDNN, one epoch of IMDB training takes ~250 secocnds

If you want to use `CuDNN`, you really should also

```
[dnn]
conv.algo_fwd = time_on_shape_change
conv.algo_bwd = time_on_shape_change
```
to your `.theanorc`.

## IMDB Dataset

Run `python prepare-imdb.py` to prepare your data!

## Yelp Humor Dataset

Run `python prepare-yelp.py` to prepare your data!

