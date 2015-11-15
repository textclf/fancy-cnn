'''
sequences.py -- utilities for sequences [WIP]
'''
def normalize_sos(sq, sz=30):
    '''
    Take a list of lists and ensure that they are all of length `sz`

    Args:
    -----

        e: a non-generator iterable of lists
    '''
    def _normalize(e, sz):
        return e[:sz] if len(e) >= sz else e + [0] * (sz - len(e))
    return [_normalize(e, sz) for e in sq]
