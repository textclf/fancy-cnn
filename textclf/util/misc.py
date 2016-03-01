from functools import wraps as DECORATOR

import language


def memoize(obj):
    '''
    A simple method decorator for memoization
    Usage
    ------
    @memoize
    def myfunction(arg1, arg2):
        return 2 * arg2 + arg1
    Internals:
    ----------
        This decorator hashes a string representation of the args+kwargs
        passed to cache preci
    '''
    # -- initialize memoization cache
    cache = obj.cache = {}

    # -- create wrapper around general object function calls
    @DECORATOR(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer

def normalize_sos(sq, sz=30, filler=0, prepend=True):
    '''
    Take a list of lists and ensure that they are all of length `sz`

    Args:
    -----
        e: a non-generator iterable of lists

        sz: integer, the size that each sublist should be normalized to

        filler: obj -- what should be added to fill out the size?

        prepend: should `filler` be added to the front or the back of the list?
        
    '''
    if not prepend:
        def _normalize(e, sz):
            return e[:sz] if len(e) >= sz else e + [filler] * (sz - len(e))
        return [_normalize(e, sz) for e in sq]
    else:
        def _normalize(e, sz):
            return e[-sz:] if len(e) >= sz else [filler] * (sz - len(e)) + e
        return [_normalize(e, sz) for e in sq]


def to_glove_vectors(text, glovebox):
    tokens = language.tokenize_text(text)

    wvs = []
    for token in tokens:
        wvs.append(glovebox[token])

    return wvs