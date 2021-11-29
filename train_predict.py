#!/usr/bin/env python3

import argparse
import os
import sys
import traceback
import warnings
import json
import collections
import pickle
import joblib

import confidence
import lir
from lir import transformers
import numpy as np
import scipy.spatial
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
import sklearn.model_selection
import sklearn.neighbors
import sklearn.pipeline
import sklearn.preprocessing


from authorship import fisher_data
from authorship import roxsd_data
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


def get_batch_simple(X, y, conv_ids, repeats, pairs=None, max_n_of_pairs_per_class=None):
    for i in range(repeats):
        authors = np.unique(y)
        authors_train, authors_test = sklearn.model_selection.train_test_split(authors, test_size=.71, random_state=i)

        X_sub_train = X[np.isin(y, authors_train), :]
        y_sub_train = y[np.isin(y, authors_train)]
        conv_ids_sub_train = conv_ids[np.isin(y, authors_train)]

        X_train, y_train, conv_train = get_pairs(X_sub_train, y_sub_train, conv_ids_sub_train, pairs,
                                                 max_n_of_pairs_per_class)

        X_sub_test = X[np.isin(y, authors_test), :]
        y_sub_test = y[np.isin(y, authors_test)]
        conv_ids_sub_test = conv_ids[np.isin(y, authors_test)]

        X_test, y_test, conv_test = get_pairs(X_sub_test, y_sub_test, conv_ids_sub_test, pairs,
                                              max_n_of_pairs_per_class)

        print('train same: ', int(np.sum(y_train)))
        print('train diff: ', int(y_train.size - np.sum(y_train)))
        print('test same: ', int(np.sum(y_test)))
        print('test diff: ', int(y_test.size - np.sum(y_test)))

        yield X_train, y_train, conv_train, X_test, y_test, conv_test


def get_pairs(X, y, conv_ids, pairs=None, sample_size=None):
    # pair instances: same source and different source

    if pairs is None:
        if sample_size is None:
            pairs_transformation = transformers.InstancePairing(same_source_limit=None,
                                                                different_source_limit='balanced')
        else:
            pairs_transformation = transformers.InstancePairing(same_source_limit=sample_size,
                                                                different_source_limit=sample_size)
        X_pair, y_pair = pairs_transformation.transform(X, y)
        pairing = pairs_transformation.pairing  # indices of pairs based on the transcriptions
        conv_pairs = np.apply_along_axis(lambda a: np.array([conv_ids[a[0]], conv_ids[a[1]]]), 1, pairing)
    else:
        with open(pairs, 'rt') as f:
            predefined_pairs = [[token for token in line.split() if 'target' not in token] for line in f]
        predefined_pairs = [set(pair) for pair in predefined_pairs]

        # pair instances: same source and different source
        pairs_transformation = transformers.InstancePairing(same_source_limit=None, different_source_limit=None)
        X_all_pairs, y_all_pairs = pairs_transformation.transform(X, y)
        pairing = pairs_transformation.pairing  # indices of pairs based on the transcriptions
        # from indices to the actual pairs
        conv_pairs = []
        for pair in pairing:
            conv_pairs.append([conv_ids[pair[0]], conv_ids[pair[1]]])
        conv_pairs = np.array(conv_pairs)

        in_predefined = np.apply_along_axis(lambda a: True if set(a) in predefined_pairs else False, 1, conv_pairs)

        X_pair = X_all_pairs[in_predefined, :, :]
        y_pair = y_all_pairs[in_predefined]
        conv_pairs = conv_pairs[in_predefined]

    return X_pair, y_pair, conv_pairs


def train_and_predict(desc, dataset_train, dataset_val, n_frequent_words, max_n_of_pairs_per_class, preprocessor,
                      classifier, calibrator, resultdir, extra_file_train=None, pairsdir_val=None,
                      min_words_in_conv=50, repeats=1):
    """
    TODO: update the description
    Train on train set and apply on validation set

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

    ds_train = fisher_data.FisherDataSource(dataset_train, extra_file_train, n_frequent_words=n_frequent_words,
                                      min_words_in_conv=min_words_in_conv)
    X, y = ds_train.get()

    assert X.shape[0] > 0

    desc_pre = '_'.join(name for name, tr in preprocessor.steps).replace(':', '_')
    desc_clf = '_'.join(name for name, tr in clf.scorer.steps).replace(':', '_')
    folder_name = f'n_freq_words={n_frequent_words}_{desc_pre}_{desc_clf}'
    path = os.path.join(resultdir, folder_name)
    os.makedirs(path, exist_ok=True)

    with open(f'{path}/wordlist.json', 'w', encoding='utf-8') as f:
        json.dump(ds_train.wordlist, f, indent=4)

    X = preprocessor.fit_transform(X)

    X, y = transformers.InstancePairing(same_source_limit=max_n_of_pairs_per_class,
                                        different_source_limit=max_n_of_pairs_per_class).transform(X, y)
    clf.fit(X, y)

    # load and prep roxsd_data
    ds_val = roxsd_data.RoxsdDataSource(dataset_val, path, min_words_in_conv=min_words_in_conv)
    X_val, y_val, conv_ids_val = ds_val.get()

    assert X_val.shape[0] > 0

    X_val = preprocessor.transform(X_val)

    if repeats == 1:
        X_pairs, y_pairs, conv_pairs = get_pairs(X_val, y_val, conv_ids_val, pairsdir_val)
        lrs = clf.predict_lr(X_pairs)
    if repeats > 1:
        lrs = []
        y_all = []
        spec_cal = lir.ScalingCalibrator(lir.KDECalibrator())
        for X_train, y_train, conv_train, X_test, y_test, conv_test in tqdm(get_batch_simple(X_val, y_val, conv_ids_val,
                                                                                             repeats, pairsdir_val)):
            scores_train = lir.apply_scorer(clf.scorer, X_train)
            spec_cal.fit(X=scores_train, y=y_train)
            scores_test = lir.apply_scorer(clf.scorer, X_test)
            lrs.append(spec_cal.transform(scores_test))
            y_all.append(y_test)
        lrs = np.concatenate(lrs)
        y_pairs = np.concatenate(y_all)


    # with lir.plotting.savefig(f'{resultdir}/pav.png') as ax:
    #     ax.pav(lrs, y_pairs)

    cllr = lir.metrics.cllr(lrs, y_pairs)
    cllr_min = lir.metrics.cllr_min(lrs, y_pairs)  # discrimination power
    acc = np.mean((lrs > 1) == y_pairs)
    recall = np.mean(lrs[y_pairs == 1] > 1)  # true positive rate
    precision = np.mean(y_pairs[lrs > 1] == 1)
    tnr = np.mean(lrs[y_pairs == 0] <= 1)  # true negative rate

    fpr, tpr, threshold = roc_curve(list(y_pairs), list(lrs), pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    auc = roc_auc_score(list(y_pairs), list(lrs))
    Metrics = collections.namedtuple('Metrics', ['cllr', 'accuracy', 'eer', 'auc', 'recall', 'precision', 'tnr', 'cllr_min'])
    results = Metrics(cllr, acc, eer, auc, recall, precision, tnr, cllr_min)

    print(f'{desc}: {results._fields} = {list(np.round(results, 3))}')

    return results


def run(dataset_train, dataset_val, resultdir, extra_file_train=None, pairsdir_val=None):
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

    br_mlp = sklearn.pipeline.Pipeline([
        ('diff:bray', BrayDistance()),
        ('clf:mlp', MLPClassifier(solver='adam', max_iter=800, alpha=0.001, hidden_layer_sizes=(5, 10, 2), random_state=1)),
    ])

    man_logit = sklearn.pipeline.Pipeline([
        ('diff:abs', transformers.AbsDiffTransformer()),
        ('clf:logit', LogisticRegression(max_iter=600, class_weight='balanced')),
    ])

    exp = experiments.Evaluation(train_and_predict)

    exp.parameter('dataset_train', dataset_train)
    exp.parameter('dataset_val', dataset_val)
    exp.parameter('extra_file_train', extra_file_train)
    exp.parameter('pairsdir_val', pairsdir_val)
    exp.parameter('resultdir', resultdir)

    exp.parameter('min_words_in_conv', 10)
    exp.parameter('n_frequent_words', 100)
    exp.parameter('max_n_of_pairs_per_class', 20000)
    exp.parameter('repeats', 1)

    exp.parameter('preprocessor', prep_gauss)
    exp.parameter('classifier', ('br_logit', br_logit))
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
    parser.add_argument('--data_val', metavar='VALDATAFILENAME',
                        help=f'dataset to be used for validation val; index file as generated by `sha256sum` '
                             f'(default: {config.data_val})',
                        default=config.data_val)
    parser.add_argument('--pairs_directory', metavar='PAIRSDIRNAME',
                        help=f'predefined pairs` (default: {config.pairsdir})',
                        default=config.pairsdir)
    parser.add_argument('--output-directory', '-o', metavar='DIRNAME',
                        help=f'path to generated output files (default: {config.resultdir})', default=config.resultdir)
    args = parser.parse_args()

    os.makedirs(args.output_directory, exist_ok=True)
    run(args.data, args.data_val, args.output_directory, args.extra_data_file, args.pairs_directory)
