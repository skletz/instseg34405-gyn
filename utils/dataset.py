import sklearn
import pickle
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from skmultilearn.model_selection import IterativeStratification
import numpy as np
from typing import List, Tuple, Optional


class DatasetUtil(object):

    @staticmethod
    def train_val_test_split(X, y, train_size, random_state=None):
        """
        Randomly split dataset, based on these ratios:
            'train': train_size
            'validation': (1-train_size) / 2
            'test':  (1-train_size) / 2

        Example: train_size=0.6 gives a 60% / 20% / 20% split

        :param X: (list) Source data to be split
        :param y: (list) Target data to be split
        :param train_size: (float) proportion in range between 0.0 and 1.0
        :param shuffle: (boolean, default=false) proportion in range between 0.0 and 1.0
        :param random_state: (int, RandomState or None, default=None) if None then np.random is used as instance
        :return: (lists) X_train, X_val, X_test, Y_train, Y_val, Y_test
        """

        assert train_size >= 0 and train_size <= 1, "Invalid training set size"

        sampler = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
        sample_indices = list(sampler.split(X, y))

        X_tmp, y_tmp = DatasetUtil.get_subsets(X, y, sample_indices[0][1])

        X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(
            X_tmp, y_tmp, train_size=0.5, random_state=random_state, shuffle=True, stratify=y_tmp)

        val_indices = []
        for i in X_val:
            index = X.index(i)
            val_indices.append(index)

        test_indices = []
        for i in X_test:
            index = X.index(i)
            test_indices.append(index)

        sample_indices[0] = (sample_indices[0][0], np.array(val_indices), np.array(test_indices))

        return sample_indices

    @staticmethod
    def train_val_test_split_multilabel(X: List, y: List, train_size: int, random_state: Optional[int] = None) -> List[
        Tuple[np.array, np.array, np.array]]:
        """

        :param X:
        :param y:
        :param train_size:
        :param random_state:
        :return: list of tuples, one tuple represents one split with train, val, test indices
        each split is a ndarray of indices in X
        """
        # logging.debug("Sample size %d" % len(X))

        assert train_size >= 0 and train_size <= 1, "Invalid training set size"

        X_cp = X
        y_cp = y

        X = np.asarray(X)
        y = np.asarray(y)

        test_size = 1.0 - train_size
        # logging.debug("Train Size %d" % (len(X) * train_size))
        # logging.debug("Sample size %d" % (len(X) * test_size))

        stratifier = IterativeStratification(n_splits=2, order=1,
                                             sample_distribution_per_fold=[test_size, 1.0 - test_size],
                                             random_state=random_state)
        train_indexes, test_indexes = next(stratifier.split(X, y))

        # print("Train Size", train_indexes.shape)

        X_train = X[train_indexes]
        y_train = y[train_indexes, :]
        X_test = X[test_indexes]
        y_test = y[test_indexes, :]

        stratifier = IterativeStratification(n_splits=2, order=1,
                                             sample_distribution_per_fold=[0.5, 0.5],
                                             random_state=random_state)

        test_indexes, val_indexes = next(stratifier.split(X_test, y_test))

        X_val = X_test[val_indexes]
        y_val = y_test[val_indexes, :]

        X_test = X_test[test_indexes]
        y_test = y_test[test_indexes, :]

        val_indices = []
        for i in X_val:
            index = X_cp.index(i)
            val_indices.append(index)

        test_indices = []
        for i in X_test:
            index = X_cp.index(i)
            test_indices.append(index)

        # print("Val Size", val_indexes.shape)
        # print("Test Size", test_indexes.shape)

        sample_indices = []
        sample_indices.append((train_indexes, np.array(val_indices), np.array(test_indices)))

        return sample_indices


    @staticmethod
    def get_subsets(X, y, index):
        subset_X = [X[i] for i in index]
        subset_y = [y[i] for i in index]
        return subset_X, subset_y

    @staticmethod
    def save_samples_split_indices(sample_split_indices, split_names, path):
        concat_str = "_".join(split_names)
        filename = "{}_split.pkl".format(concat_str)
        file = os.path.join(path, filename)
        pickle.dump(sample_split_indices, open(file, 'wb'), pickle.HIGHEST_PROTOCOL)
        pass

    @staticmethod
    def save_dataset(dataset, path):
        file = os.path.join(path, "dataset.pkl")
        pickle.dump(dataset, open(file, 'wb'), pickle.HIGHEST_PROTOCOL)
        pass

    @staticmethod
    def restore_samples_split_indices(restore_file):

        file = restore_file
        sample_split_indices = pickle.load(open(file, 'rb'))
        return sample_split_indices

    @staticmethod
    def restore_dataset(path):
        file = os.path.join(path, "dataset.pkl")
        dataset = pickle.load(open(file, 'rb'))
        return dataset
