try:
    from spacy.en import English
except ImportError:
    print '[!] You need to install spaCy! Visit spacy.io/#install'

# Spacy.en provides a faster tokenizer than nltk
nlp = English()

def parse_paragraph(txt):
    """
    Takes a text and returns a list of lists of tokens, where each sublist is a sentence
    """
    sentences = nlp(u'' + txt.decode('ascii', errors='ignore')).sents
    return [[t.text for t in s] for s in sentences]

def tokenize_text(text):
    """
    Gets tokens from a text in English
    """
    if not isinstance(text, unicode):
        text = unicode(text)

    tokens = [token.lower_ for token in nlp(text)]

    return tokens

def _calculate_languages_ratios(text):
    """
    Calculate probability of given text to be written in several languages and
    return a dictionary that looks like {'french': 2, 'spanish': 4, 'english': 0}

    @param text: Text whose language want to be detected
    @type text: str

    @return: Dictionary with languages and unique stopwords seen in analyzed text
    @rtype: dict
    """

    languages_ratios = {}
    tokens = tokenize_text(text)

    # Compute per language included in nltk number of unique stopwords appearing in analyzed text
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(tokens)
        common_elements = words_set.intersection(stopwords_set)

        languages_ratios[language] = len(common_elements) # language "score"

    return languages_ratios

def detect_language(text):
    """
    Calculate probability of given text to be written in several languages and
    return the highest scored.

    It uses a stopwords based approach, counting how many unique stopwords
    are seen in analyzed text.

    @param text: Text whose language want to be detected
    @type text: str

    @return: Most scored language guessed
    @rtype: str
    """
    ratios = _calculate_languages_ratios(text)
    most_rated_language = max(ratios, key=ratios.get)

    return most_rated_language
