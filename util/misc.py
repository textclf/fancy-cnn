import language

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