from base_handler import BaseDataHandler, DataHandlerException

import glob
import os

class ImdbDataHandler(BaseDataHandler):
    """
    Works with the original Large Movie Review Dataset - IMDB data as downloaded from
    http://ai.stanford.edu/~amaas/data/sentiment/

    source defines the folder where the data is downloaded
    """

    def get_data(self, type=BaseDataHandler.DATA_TRAIN):
        """
        Process the data from its source and returns two lists: texts and labels, ready for a classifier to be used

        Data is not shuffled
        """
        if type not in (BaseDataHandler.DATA_TRAIN, BaseDataHandler.DATA_TEST):
            raise DataHandlerException("Only train and test data supported for ImdbHandler")
        else:
            which_data = 'train' if type == BaseDataHandler.DATA_TRAIN else 'test'

        positive_examples = glob.glob(os.path.join(self.source, which_data, 'pos', '*.txt'))
        negative_examples = glob.glob(os.path.join(self.source, which_data, 'neg', '*.txt'))

        data = []
        labels = []
        for i, f in enumerate(positive_examples):
            data.append((open(f, 'rb').read().lower()).replace('<br /><br />', '\n'))
            labels.append(1)
        for i, f in enumerate(negative_examples):
            data.append((open(f, 'rb').read().lower()).replace('<br /><br />', '\n'))
            labels.append(0)

        return (data, labels)

