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
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from time import strftime

class TextClassifierException(Exception):
    pass

class TextClassifier(object):

    def __init__(self, train_texts=None, train_labels=None,
                 compute_features=False,
                 train=False, ngram_range=(1,1),
                 test_texts=None, test_labels=None):

        self.ngram_range = ngram_range
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.test_texts = test_texts
        self.test_labels = test_labels

        # Feature vectors: count and tfidf transformers
        self.count_vect = None
        self.tfidf_transformer = None
        # Classifier
        self.clf = None

        if compute_features:
            self.set_bag_of_ngrams()
        if train:
            self.train()

    def set_training_data(self, train_texts, train_labels, compute_features=True):
        self.train_texts = train_texts
        self.train_labels = train_labels

        if compute_features:
            self.set_bag_of_ngrams()
        else:
            # Feature vectors are not valid anymore
            self.count_vect = None
            self.tfidf_transformer = None
        self.clf = None

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

        return precision_recall_fscore_support(labels, predicted, average='macro')

    def _grid_search(self, estimator, param_grid=None, scoring=None,
                     n_jobs=1, cv=None, verbose=0):
        """
        See http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html
        for description on parameters

        Note: Watch out scoring function for unbalanced problems
        """

        print strftime("%Y-%m-%d %H:%M:%S") + ": Starting grid search for " + self.__class__.__name__ + \
              ". n-gram range = " + str(self.ngram_range)

        self.clf = GridSearchCV(estimator, param_grid=param_grid, scoring=scoring,
                                n_jobs=n_jobs, cv=cv, verbose=verbose)
        X = self._to_features(self.train_texts)
        self.clf.fit(X, self.train_labels)

        print strftime("%Y-%m-%d %H:%M:%S") + ": Finished grid search for " + self.__class__.__name__
        print "Best error rate: " + str(1 - self.clf.best_score_)
        print "Best params: " + str(self.clf.best_params_)


class NaiveBayesClassifier(TextClassifier):
    """
    Naive bayes classifier for text classification
    """

    def train(self):
        self._train_checks()
        X = self._to_features(self.train_texts)
        self.clf = MultinomialNB().fit(X, self.train_labels)

    def grid_search_cv(self, param_grid=None, scoring=None,
                       n_jobs=1, cv=None, verbose=0):
        if param_grid is None:
            # Some default nice parameters
            param_grid={'alpha': [0.1, 0.5, 1, 2, 3]}

        self._grid_search(MultinomialNB(), param_grid, scoring, n_jobs, cv, verbose)

class SGDTextClassifier(TextClassifier):
    """
    Stochastic Gradient Descent for text classification
    """
    def train(self, loss="log", alpha=0.001, penalty="l2", n_iter=5):
        self._train_checks()
        X = self._to_features(self.train_texts)
        self.clf = SGDClassifier(loss=loss, alpha=alpha,
                                 penalty=penalty,
                                 n_iter=n_iter).fit(X, self.train_labels)

    def grid_search_cv(self, param_grid=None, scoring=None,
                       n_jobs=1, cv=None, verbose=0):
        if param_grid is None:
            # Some default nice parameters
            param_grid={'alpha': 10.0**-np.arange(1,6),
                        'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge',
                                 'perceptron','squared_loss','huber',
                                 'epsilon_insensitive','squared_epsilon_insensitive'],
                        'penalty': ['none', 'l2', 'l1', 'elasticnet']}

        self._grid_search(SGDClassifier(), param_grid, scoring, n_jobs, cv, verbose)

class LogisticClassifier(TextClassifier):
    """
    Logistic Regression for text classification
    """

    def train(self, C=1.0, penalty="l2"):
        self._train_checks()
        X = self._to_features(self.train_texts)
        self.clf = LogisticRegression(C=C, penalty=penalty).fit(X, self.train_labels)

    def grid_search_cv(self, param_grid=None, scoring=None,
                       n_jobs=1, cv=None, verbose=0):
        if param_grid is None:
            # Some default nice parameters
            param_grid={'penalty': ['l1', 'l2'],
                        'C': [1.0, 0.5, 0.25, 0.0125, 0.001, 0.0001, 5, 10, 50]}

        self._grid_search(LogisticRegression(), param_grid, scoring, n_jobs, cv, verbose)

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
