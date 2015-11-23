"""
Compute WordVectors using Yelp Data
"""
from gensim.models.word2vec import Word2Vec
from util.language import detect_language, tokenize_text
from data_handling import get_reviews_data

# Set to true for zero in in English reviews. Makes the process much slower
FILTER_ENGLISH = True
# Name for output w2v model file
OUTPUT_MODEL_FILE = "w2v_yelp_100_alpha_0.025_window_4"
PICKLED_DATA = "/home/alfredo/deep-nlp/data/reviews.pickle."

NUM_PARTITIONS = 2 # Use all data
reviews_texts, _, _, _, _ = get_reviews_data(range(1, NUM_PARTITIONS), PICKLED_DATA)

# Each review will be considered a sentence
sentences = []
for num, text in enumerate(reviews_texts):
    if num % 10000 == 0:
        print "%d out of %d reviews read" % (num, len(reviews_texts))
    if FILTER_ENGLISH:
        if detect_language(text) == u"english":
            sentences.append(tokenize_text(text))
    else:
        sentences.append(text)

# Build a w2v model
w2v = Word2Vec(sentences=sentences, size=100, alpha=0.025, window=4, min_count=2, sample=1e-5, workers=4, negative=10)
w2v.save(OUTPUT_MODEL_FILE)

