import spacy.en

tokenizer = spacy.en.English()

def normalize_sos(sq, sz=30):
    '''
    Take a list of lists and ensure that they are all of length `sz`

    Args:
    -----

        e: a non-generator iterable of lists
    '''
    def _normalize(e, sz):
        return e[:sz] if len(e) >= sz else e + [filler] * (sz - len(e))
    return [_normalize(e, sz) for e in sq]

def tokenize_text(text):
    if not isinstance(text, unicode):
        text = unicode(text)

    tokens = [token.lower_ for token in tokenizer(text)]

    return tokens

def to_glove_vectors(text, glovebox):
    return [glovebox[token] for token in tokenize_text(text)]




