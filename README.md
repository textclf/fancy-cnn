# `fancy-cnn`
Convolutional Neural Networks for multi-sentence sentiment analysis (Stanford CS224N)

##General Considerations
First off, this class of sequential convolutional networks is quite `GEMM` intensive, and *really* isn't suited to a CPU. You also *really* should use `CuDNN` when training. 

Bleeding edge installations of Theano and Keras are required.

For reference: 

* On a 2.6 GHz Intel Core i7, one epoch of IMDB training takes ~3.5 days
* On a GRID K520 without CuDNN, one epoch of IMDB training takes ~1 hour
* On a GRID K520 *with* CuDNN, one epoch of IMDB training takes ~30 minutes
* On a GTX Titan X with CuDNN, one epoch of IMDB training takes ~11 minutes

If you want to use `CuDNN`, you really should also

```
[dnn]
conv.algo_fwd = time_on_shape_change
conv.algo_bwd = time_on_shape_change
```
to your `.theanorc`.

## Setup

1. Go to `datasets/aclImdb/`
2. `./download.sh`
3. Go to `embeddings/`
4. `./download.sh`

This gets all the embeddings from `S3` and sets up the IMDB dataset for processing. Yelp instructions forthcoming.

## IMDB Dataset

Run `python prepare-imdb.py` to prepare your data! Look at [this](https://github.com/lukedeo/fancy-cnn/blob/master/multichannel-imdb.py) to see how to train the IMDB model.

## Yelp Humor Dataset

Run `python prepare-yelp.py` to prepare your data! Look at [this](https://github.com/lukedeo/fancy-cnn/blob/master/model-yelp.py) to see how to train a model on Yelp.

