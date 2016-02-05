from classic.classifiers import TextClassifier, NaiveBayesClassifier, SGDTextClassifier, \
    LogisticClassifier, SVMClassifier, PerceptronClassifier, RandomForestTextClassifier
from datahandlers import ImdbDataHandler

IMDB_DATA = './datasets/aclImdb/aclImdb'

if __name__ == '__main__':

    print "Loading data from original source"
    imdb = ImdbDataHandler(source=IMDB_DATA)
    (train_reviews, train_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TRAIN)
    (test_reviews, test_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TEST)
    # TODO: Shuffle data

    # Simple bag of words with SGD
    sgd = SGDTextClassifier(train_reviews, train_labels,
                            test_texts=test_reviews, test_labels=test_labels,
                            compute_features=True)
    sgd.grid_search_cv(verbose=0, n_jobs=4)

    # Simple bag of words with NB
    nb = NaiveBayesClassifier(train_reviews, train_labels,
                              test_texts=test_reviews, test_labels=test_labels)
    nb.set_bag_of_ngrams() # Also can compute bag of words manually
    nb.grid_search_cv(n_jobs=4)

    # Now shit with bigrams too
    sgd = SGDTextClassifier(train_reviews, train_labels, ngram_range=(1,2),
                            test_texts=test_reviews, test_labels=test_labels,
                            compute_features=True)
    sgd.grid_search_cv(n_jobs=4, verbose=1)

    nb = NaiveBayesClassifier(train_reviews, train_labels, ngram_range=(1,2),
                            test_texts=test_reviews, test_labels=test_labels,
                            compute_features=True)
    nb.grid_search_cv(n_jobs=4, verbose=1)

    lr = LogisticClassifier(train_reviews, train_labels, ngram_range=(1,2),
                            test_texts=test_reviews, test_labels=test_labels,
                            compute_features=True)
    lr.grid_search_cv(verbose=5, n_jobs=4)

    # print "Naive Bayes"
    # nb = NaiveBayesClassifier()
    # nb.set_training_data(train_reviews, train_labels)
    # nb.set_test_data(test_reviews, test_labels)
    # nb.set_bag_of_ngrams()
    #
    # nb.train()
    # train_error = nb.get_training_error()
    # test_error = nb.get_test_error()
    # print "Training error: " + str(train_error)
    # print "Test error: " + str(test_error)

    # print "SGD Classifier"
    # sgd = SGDTextClassifier(train_reviews, train_labels,
    #                         test_texts=test_reviews, test_labels=test_labels)
    # #train_error = sgd.get_training_error()
    # #test_error = sgd.get_test_error()
    # #print "Training error: " + str(train_error)
    # #print "Test error: " + str(test_error)
    # sgd.set_bag_of_ngrams()
    # sgd.grid_search_cv(verbose=0, n_jobs=4)


    # print "Logistic classifier"
    # sgd = LogisticClassifier()
    # sgd.set_training_data(train_reviews, train_labels)
    # sgd.set_test_data(test_reviews, test_labels)
    # sgd.set_bag_of_ngrams()
    #
    # sgd.train()
    # train_error = sgd.get_training_error()
    # test_error = sgd.get_test_error()
    # print "Training error: " + str(train_error)
    # print "Test error: " + str(test_error)
    #
    # print "SVM classifier"
    # sgd = SVMClassifier()
    # sgd.set_training_data(train_reviews, train_labels)
    # sgd.set_test_data(test_reviews, test_labels)
    # sgd.set_bag_of_ngrams()
    #
    # sgd.train()
    # train_error = sgd.get_training_error()
    # test_error = sgd.get_test_error()
    # print "Training error: " + str(train_error)
    # print "Test error: " + str(test_error)
    #
    # print "Perceptron classifier"
    # sgd = PerceptronClassifier()
    # sgd.set_training_data(train_reviews, train_labels)
    # sgd.set_test_data(test_reviews, test_labels)
    # sgd.set_bag_of_ngrams()
    #
    # sgd.train()
    # train_error = sgd.get_training_error()
    # test_error = sgd.get_test_error()
    # print "Training error: " + str(train_error)
    # print "Test error: " + str(test_error)
    #
    # print "Random forest classifier"
    # sgd = RandomForestTextClassifier()
    # sgd.set_training_data(train_reviews, train_labels)
    # sgd.set_test_data(test_reviews, test_labels)
    # sgd.set_bag_of_ngrams()
    #
    # sgd.train()
    # train_error = sgd.get_training_error()
    # test_error = sgd.get_test_error()
    # print "Training error: " + str(train_error)
    # print "Test error: " + str(test_error)