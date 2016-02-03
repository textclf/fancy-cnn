from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm):
    """
    Plots a scikit learn confusion matrix
    """
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True category')
    plt.xlabel('Predicted category')
    plt.show()

class TextClassifierException(Exception):
    pass

class TextClassifier(object):

    def __init__(self, ngram_range=(1,1),
                 train_texts=None, train_labels=None,
                 validation_texts=None, validation_labels=None,
                 test_texts=None, test_labels=None):

        self.ngram_range = ngram_range
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.validation_texts = validation_texts
        self.validation_labels = validation_labels
        self.test_texts = test_texts
        self.test_labels = test_labels

        # Feature vectors: count and tfidf transformers
        self.count_vect = None
        self.tfidf_transformer = None
        # Classifier
        self.clf = None

    def set_training_data(self, train_texts, train_labels):
        self.train_texts = train_texts
        self.train_labels = train_labels

        # Feature vectors are not valid anymore
        self.count_vect = None
        self.tfidf_transformer = None

    def set_validation_data(self, validation_texts, validation_labels):
        self.validation_texts = validation_texts
        self.validation_labels = validation_labels

    def set_test_data(self, test_texts, test_labels):
        self.test_texts = test_texts
        self.test_labels = test_labels

    def set_ngram_range(self, ngram_range):
        self.ngram_range = ngram_range

    def set_bag_of_ngrams(self):
        """ Sets vectorizer feature vector based on the training data"""
        self.count_vect = CountVectorizer(ngram_range=self.ngram_range,
                                          stop_words="english")
        X_train_counts = self.count_vect.fit_transform(self.train_texts)
        self.tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)

    def _to_features(self, texts):
        if self.count_vect is None:
            raise TextClassifierException("Feature transformer not defined. Call set_bag_of_ngrams")
        texts_counts = self.count_vect.transform(texts)
        texts_tfidf = self.tfidf_transformer.transform(texts_counts)
        return texts_tfidf

    def train(self):
        raise NotImplementedError()

    def _train_checks(self):
        if self.train_texts is None or self.train_labels is None:
            raise TextClassifierException("No complete train data found")
        if self.count_vect is None or self.tfidf_transformer is None:
            raise TextClassifierException("Feature vectors not computed")

    def __test(self, texts, labels):
        X = self._to_features(texts)
        predicted = self.clf.predict(X)
        return 1 - np.mean(predicted == labels)

    def get_training_error(self):
        if self.clf is None:
            raise TextClassifierException("No classifier was trained")
        return self.__test(self.train_texts, self.train_labels)

    def get_test_error(self):
        if self.clf is None:
            raise TextClassifierException("No classifier was trained")
        return self.__test(self.test_texts, self.test_labels)

    def get_scores(self, texts=None, labels=None):
        """
        Returns (precision, recall, f_score, support)
        Defaults to test loaded test data
        """
        if texts is None or labels is None:
            texts = self.test_texts
            labels = self.test_labels
        X = self._to_features(texts)
        predicted = self.clf.predict(X)
        self.cm = confusion_matrix(labels, predicted)

        return precision_recall_fscore_support(labels, predicted, average='macro')


class NaiveBayesClassifier(TextClassifier):
    """
    Naive bayes classifier for text classification
    """

    def train(self):
        self._train_checks()
        X = self._to_features(self.train_texts)
        self.clf = MultinomialNB().fit(X, self.train_labels)

class SGDTextClassifier(TextClassifier):
    """
    Stochastic Gradient Descent for text classification
    """
    def train(self):
        self._train_checks()
        X = self._to_features(self.train_texts)
        self.clf = SGDClassifier(loss="modified_huber", alpha=0.001,
                                 penalty="l2").fit(X, self.train_labels)

class LogisticClassifier(TextClassifier):
    """
    Logistic Regression for text classification
    """

    def train(self):
        self._train_checks()
        X = self._to_features(self.train_texts)
        self.clf = LogisticRegression(C=1e5).fit(X, self.train_labels)

class SVMClassifier(TextClassifier):
    """
    SVM for text classification
    """
    def train(self):
        self._train_checks()
        X = self._to_features(self.train_texts)
        self.clf = LinearSVC(C=0.4).fit(X, self.train_labels)

class PerceptronClassifier(TextClassifier):
    """
    Perceptron Classifier for text classification
    """
    def train(self):
        self._train_checks()
        X = self._to_features(self.train_texts)
        self.clf = Perceptron().fit(X, self.train_labels)

class RandomForestTextClassifier(TextClassifier):
    """
    K nearest neighbors for text classification (probably worst idea ever)
    """
    def train(self):
        self._train_checks()
        X = self._to_features(self.train_texts)
        self.clf = RandomForestClassifier().fit(X, self.train_labels)
