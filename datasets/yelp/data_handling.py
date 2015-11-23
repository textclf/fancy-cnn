"""
Functions to read and extract review data from the Yelp Dataset Challenge dataset.
"""

import json
import pickle
import random
import sys
import numpy as np

DEFAULT_REVIEWS_FILE = "data/yelp_academic_dataset_review.json"
DEFAULT_REVIEWS_PICKLE = "data/reviews.pickle"

def pickles_from_json(json_file=DEFAULT_REVIEWS_FILE, pickle_name=DEFAULT_REVIEWS_PICKLE, num_partitions=100,
                      accepted=None):
    """
    Dumps a json into a number of pickle partitions, which contain a list of python objects.

    accepted is a generic function that returns true or false for a single json object, specifying whether or not
    the object should be added to the pickle
    """

    print "Reading json file..."
    object = []
    num_not_accepted = 0
    total_processed = 0
    with open(json_file) as json_data:
        for line in json_data:
            if accepted != None:
                element = json.loads(line)
                if accepted(element):
                    object.append(element)
                else:
                    num_not_accepted += 1
                    sys.stdout.write('Not accepted objects: %d / %d \r' % (num_not_accepted, total_processed))
                    sys.stdout.flush()
            else:
                object.append(json.loads(line))
            total_processed += 1

    print "Shuffling resulting python objects"
    random.shuffle(object)

    length_partition = len(object)/num_partitions
    remaining_to_process = len(object)
    current_partition = 1
    while remaining_to_process > 0:
        sys.stdout.write('Importing package %d out of %d \r' % (current_partition, num_partitions))
        sys.stdout.flush()

        # All the remaining elements go to the last partition
        if current_partition == num_partitions:
            stop = None
            num_in_partition = remaining_to_process
        else:
            stop = -remaining_to_process + length_partition
            num_in_partition = length_partition

        pickle.dump(object[-remaining_to_process:stop],
                    open(pickle_name + '.' + str(current_partition), "wb"),
                    pickle.HIGHEST_PROTOCOL)

        current_partition += 1
        remaining_to_process -= num_in_partition

def load_partitions(partition_list, pickle_base_name=DEFAULT_REVIEWS_PICKLE + '.'):
    """
    Returns a python object being a list of dictionaries.
    It reads the data from a sequence of files starting with the given base name. For instance:
    partition_list = [2,4,6], pickle_base_name = "pickle." will read files pickle.2, pickle.4, pickle.6
    """

    num_partition = 1
    result = []
    for partition in partition_list:
        print 'Reading partition %d of %d' % (num_partition, len(partition_list))
        with open(pickle_base_name + str(partition)) as file:
            loaded_element = pickle.load(file)
            result.extend(loaded_element)

        num_partition += 1

    print "Read a total of %d partitions for a total of %d objects" % (num_partition - 1, len(result))
    return result

def get_reviews_data(partitions_to_use, pickle_base_name):
    """
    Gets loaded json data in pickles and returns fields of interest
    """

    data = load_partitions(partitions_to_use, pickle_base_name)
    review_texts = []
    useful_votes = []
    funny_votes = []
    cool_votes = []
    review_stars = []

    for review in data:
        review_texts.append(review['text'])
        useful_votes.append(review['votes']['useful'])
        cool_votes.append(review['votes']['cool'])
        funny_votes.append(review['votes']['funny'])
        review_stars.append(review['stars'])

    return review_texts, useful_votes, funny_votes, cool_votes, review_stars

def give_balanced_classes(reviews, funny_votes):
    """
    From all the reviews and votes given, partitions the data into two classes: funny reviews and not
    funny reviews.
    All the funny reviews found are returned. The method is assuming majority of not funny votes.
    The same number of not funny reviews are returned, randomly selected.
    Returned data is a shuffled balanced set of funny and not funny reviews.
    """

    # We will consider a review to be funny if it has 3 or more funny votes.
    # Not funny reviews have 0 votes.
    VOTES_THRESHOLD = 3
    not_funny_reviews_indices = []

    # Find all the funny reviews we can
    final_reviews = []
    final_labels = []
    for i, review in enumerate(reviews):
        if funny_votes[i] >= VOTES_THRESHOLD:
            final_reviews.append(review)
            final_labels.append(1)
        elif funny_votes[i] == 0:
            not_funny_reviews_indices.append(i)

    # We want balanced classes so take same number
    np.random.shuffle(not_funny_reviews_indices)
    num_funny_reviews = len(final_reviews)
    for i in range(num_funny_reviews):
        final_reviews.append(reviews[not_funny_reviews_indices[i]])
        final_labels.append(0)

    # Shuffle final reviews and labels
    combined_lists = zip(final_reviews, final_labels)
    np.random.shuffle(combined_lists)
    final_reviews[:], final_labels[:] = zip(*combined_lists)

    print "Returning %d funny reviews and a total of %d reviews" % (num_funny_reviews, len(final_reviews))

    return (final_reviews, final_labels)

def create_data_sets(partition_list=range(1,100), pickle_base_name=DEFAULT_REVIEWS_PICKLE + '.'):
    """
    Creates a 50% - 25% - 25% train/validation/test partition of the classification problem. Classes are balanced.
    It reads the list of partitions saved in pickles.
    Resulting data sets are saved as python pickles.
    """

    load_partitions(partition_list, pickle_base_name)
    reviews, _, funny_votes, _, _ = get_reviews_data(partition_list, pickle_base_name)
    reviews, labels = give_balanced_classes(reviews, funny_votes)
    N = len(reviews)

    train_reviews = reviews[:N/2]
    train_labels = labels[:N/2]

    dev_reviews = reviews[N/2:3*N/4]
    dev_labels = labels[N/2:3*N/4]

    test_reviews = reviews[3*N/4:]
    test_labels = labels[3*N/4:]

    pickle.dump([train_reviews, train_labels],
                open("TrainSet_" + str(N), "wb"), pickle.HIGHEST_PROTOCOL)

    pickle.dump([dev_reviews, dev_labels],
                open("DevSet_" + str(N), "wb"), pickle.HIGHEST_PROTOCOL)

    pickle.dump([test_reviews, test_labels],
                open("TestSet_" + str(N), "wb"), pickle.HIGHEST_PROTOCOL)