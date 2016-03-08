"""
Misc functions using word vector containers
"""
import numpy as np

def get_mean_wv(wvs_container, text):
    """
    Computes the mean of all the word vectors in a text. A text is an array consistent of word indices for a given
    word vector container
    """
    vec = []
    # Append wvs to later average them
    for w in text:
        try:
            vec.append(wvs_container[u'' + wvs_container._get_i2w(w)])
        except KeyError:
            continue
    if len(vec) == 0:
        # Text with no recognized tokens, defaults to 0 vector
        vec.append(np.zeros(wvs_container.W.shape[1]))
    return np.array(vec).mean(axis=0)

def data_to_wvs(wvs_container, reviews):
    """
    Converts texts in form of wv indices ([[0, 1, 2, 3, 52, 23], [1,2,52, 23, 15]...]) into
    mean word vectors for each text
    """

    wv_size = wvs_container.W.shape[1]
    X = np.zeros((reviews.shape[0], wv_size))

    # Traverse every text and compute mean vectors
    for i in range(X.shape[0]):
        if i % 100 == 0:
            print i
        X[i,:] = get_mean_wv(wvs_container, reviews[i])

    return X
