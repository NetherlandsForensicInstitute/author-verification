#!/usr/bin/env python3

import argparse
import os
import sys
import traceback
import warnings
import json
import pickle
import joblib

import confidence
import lir
from lir import transformers
import numpy as np
import scipy.spatial
import scipy.stats
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection
import sklearn.neighbors
import sklearn.pipeline
import sklearn.preprocessing

from authorship import fisher_data
from authorship import experiments


class GaussianCdfTransformer(sklearn.base.TransformerMixin):
    def fit(self, X):
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)

        self._valid_features = self._std > 0
        self._mean = self._mean[self._valid_features]
        self._std = self._std[self._valid_features]

        return self

    def transform(self, X):
        assert len(X.shape) == 2
        X = X[:, self._valid_features]
        return scipy.stats.norm.cdf(X, self._mean, self._std)


class BrayDistance(sklearn.base.TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == 2

        left = X[:, :, 0]
        right = X[:, :, 1]

        return np.abs(right - left) / (np.abs(right + left) + 1)


def train_samesource(desc, dataset, n_frequent_words, max_n_of_pairs_per_class, preprocessor, classifier, calibrator,
                     resultdir, extra_file=None, min_words_in_conv=50):
    """
    Train and save a fit with the parameters provided.

    :param desc: free text description of the experiment
    :param dataset: path to transcript index file
    :param n_frequent_words: int: number of freq words
    :param max_n_of_pairs_per_class: int: number of pairs per class
    :param preprocessor: Pipeline: an sklearn pipeline
    :param classifier: Pipeline: an sklearn pipeline with a classifier as last element
    :param calibrator: a LIR calibrator
    :param resultdir: directory for saving 'model'
    :param extra_file: path to extra file needed to read data
    :param min_words_in_conv: min number of words a file should have for training a model
    :return: size of the data, num of ss and ds
    """

    clf = lir.CalibratedScorer(classifier, calibrator)

    ds = fisher_data.FisherDataSource(dataset, extra_file, n_frequent_words=n_frequent_words,
                                      min_words_in_conv=min_words_in_conv)
    X, y = ds.get()

    assert X.shape[0] > 0

    desc_pre = '_'.join(name for name, tr in preprocessor.steps).replace(':', '_')
    desc_clf = '_'.join(name for name, tr in clf.scorer.steps).replace(':', '_')
    folder_name = f'n_freq_words={n_frequent_words}_{desc_pre}_{desc_clf}'
    path = os.path.join(resultdir, folder_name)
    os.makedirs(path, exist_ok=True)

    with open(f'{path}/wordlist.json', 'w', encoding='utf-8') as f:
        json.dump(ds.wordlist, f, indent=4)

    X = preprocessor.fit_transform(X)

    # preprocessor - try 1
    try:
        with open(f'{path}/preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
    except Exception as e:
        print(e)
        raise

    with open(f'{path}/preprocessor.sav', 'wb') as f:
        pickle.dump(preprocessor, f)

    # preprocessor - try 2/3
    joblib.dump(preprocessor, f'{path}/preprocessor.mod')
    joblib.dump(preprocessor, f'{path}/preprocessor.joblib')

    X, y = transformers.InstancePairing(same_source_limit=max_n_of_pairs_per_class,
                                        different_source_limit=max_n_of_pairs_per_class).transform(X, y)
    clf.fit(X, y)

    # model - try 1
    with open(f'{path}/distance_classifier_calibrator.pkl', 'wb') as f:
        pickle.dump(clf, f)

    # model - try 2/3
    joblib.dump(clf, f'{path}/distance_classifier_calibrator.mod')
    joblib.dump(clf, f'{path}/distance_classifier_calibrator.joblib')

    return y.size, int(np.sum(y)), int(y.size - np.sum(y))


def run(dataset, resultdir, extra_data_file=None):
    ### PREPROCESSORS
    prep_none = sklearn.pipeline.Pipeline([
        ('scale:none', None),
        ('pop:none', None),
    ])

    prep_gauss = sklearn.pipeline.Pipeline([
        ('pop:gauss', GaussianCdfTransformer()),  # cumulative density function for each feature
    ])

    ### CLASSIFIERS
    br_logit = sklearn.pipeline.Pipeline([
        ('diff:bray', BrayDistance()),
        ('clf:logit', LogisticRegression(class_weight='balanced')),
    ])

    exp = experiments.Evaluation(train_samesource)

    exp.parameter('dataset', dataset)
    exp.parameter('extra_file', extra_data_file)
    exp.parameter('resultdir', resultdir)

    exp.parameter('min_words_in_conv', 50)
    exp.parameter('n_frequent_words', 500)
    exp.parameter('max_n_of_pairs_per_class', 15000)

    exp.parameter('preprocessor', prep_gauss)
    exp.parameter('classifier', ('bray_logit', br_logit))
    exp.parameter('calibrator', lir.ScalingCalibrator(lir.KDECalibrator()))

    try:
        exp.runDefaults()

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, file=sys.stdout)


if __name__ == '__main__':
    config = confidence.load_name('authorship_train', 'local')
    warnings.filterwarnings("error")
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', help='increases verbosity', action='count', default=0)
    parser.add_argument('-q', help='decreases verbosity', action='count', default=0)
    parser.add_argument('--data', metavar='FILENAME',
                        help=f'dataset to be used; index file as generated by `sha256sum` (default: {config.data})',
                        default=config.data)
    parser.add_argument('--extra_data_file', metavar='EXTRAFILENAME',
                        help=f'extra file needed to match speakers with conversation` (default: {config.data_info})',
                        default=config.data_info)
    parser.add_argument('--output-directory', '-o', metavar='DIRNAME',
                        help=f'path to generated output files (default: {config.resultdir})', default=config.resultdir)
    args = parser.parse_args()

    os.makedirs(args.output_directory, exist_ok=True)
    run(args.data, args.output_directory, args.extra_data_file)
