from classic.classifiers import TextClassifier, NaiveBayesClassifier, SGDTextClassifier, \
    LogisticClassifier, SVMClassifier, PerceptronClassifier, RandomForestTextClassifier
from datahandlers import ImdbDataHandler


IMDB_DATA = './datasets/aclImdb/aclImdb'

if __name__ == '__main__':

    print "Loading data from original source"
    imdb = ImdbDataHandler(source=IMDB_DATA)
    (train_reviews, train_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TRAIN)
    (test_reviews, test_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TEST)

    print "Naive Bayes"
    nb = NaiveBayesClassifier()
    nb.set_training_data(train_reviews, train_labels)
    nb.set_test_data(test_reviews, test_labels)
    nb.set_bag_of_ngrams()

    nb.train()
    train_error = nb.get_training_error()
    test_error = nb.get_test_error()
    print "Training error: " + str(train_error)
    print "Test error: " + str(test_error)

    print "SGD Classifier"
    sgd = SGDTextClassifier()
    sgd.set_training_data(train_reviews, train_labels)
    sgd.set_test_data(test_reviews, test_labels)
    sgd.set_bag_of_ngrams()

    sgd.train()
    train_error = sgd.get_training_error()
    test_error = sgd.get_test_error()
    print "Training error: " + str(train_error)
    print "Test error: " + str(test_error)

    print "Logistic classifier"
    sgd = LogisticClassifier()
    sgd.set_training_data(train_reviews, train_labels)
    sgd.set_test_data(test_reviews, test_labels)
    sgd.set_bag_of_ngrams()

    sgd.train()
    train_error = sgd.get_training_error()
    test_error = sgd.get_test_error()
    print "Training error: " + str(train_error)
    print "Test error: " + str(test_error)

    print "SVM classifier"
    sgd = SVMClassifier()
    sgd.set_training_data(train_reviews, train_labels)
    sgd.set_test_data(test_reviews, test_labels)
    sgd.set_bag_of_ngrams()

    sgd.train()
    train_error = sgd.get_training_error()
    test_error = sgd.get_test_error()
    print "Training error: " + str(train_error)
    print "Test error: " + str(test_error)

    print "Perceptron classifier"
    sgd = PerceptronClassifier()
    sgd.set_training_data(train_reviews, train_labels)
    sgd.set_test_data(test_reviews, test_labels)
    sgd.set_bag_of_ngrams()

    sgd.train()
    train_error = sgd.get_training_error()
    test_error = sgd.get_test_error()
    print "Training error: " + str(train_error)
    print "Test error: " + str(test_error)

    print "Random forest classifier"
    sgd = RandomForestTextClassifier()
    sgd.set_training_data(train_reviews, train_labels)
    sgd.set_test_data(test_reviews, test_labels)
    sgd.set_bag_of_ngrams()

    sgd.train()
    train_error = sgd.get_training_error()
    test_error = sgd.get_test_error()
    print "Training error: " + str(train_error)
    print "Test error: " + str(test_error)