# `fancy-cnn`
Convolutional Neural Networks for multi-sentence sentiment analysis (Stanford CS224N)

Run `python prepare-imdb.py`, then run `python model-test.py`! It takes ~200 seconds per epoch for a pretty basic model on a GeForce GT 750M *with* CuDNN3 and 

```
[dnn]
conv.algo_fwd = time_on_shape_change
conv.algo_bwd = time_on_shape_change
```
in your `.theanorc`.
