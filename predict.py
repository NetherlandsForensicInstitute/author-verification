import argparse
import os
import sys
import traceback
import warnings
import collections
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

from authorship import roxsd_data
from authorship import fisher_data
from authorship import experiments
from sklearn.metrics import roc_curve


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


def get_pairs(X, y, conv_ids, pairs=None):
    # pair instances: same source and different source

    if pairs is None:
        pairs_transformation = transformers.InstancePairing(same_source_limit=None, different_source_limit='balanced')
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
        conv_pairs = np.apply_along_axis(lambda a: np.array([conv_ids[a[0]], conv_ids[a[1]]]), 1, pairing)  # from indices to the actual pairs

        in_predefined = np.apply_along_axis(lambda a: True if set(a) in predefined_pairs else False, 1, conv_pairs)

        X_pair = X_all_pairs[in_predefined, :, :]
        y_pair = y_all_pairs[in_predefined]
        conv_pairs = conv_pairs[in_predefined]

    return X_pair, y_pair, conv_pairs


def predict_samesource(desc, dataset, modeldir, resultdir, pairsdir=None, min_words_in_conv=50):
    """
    Predict and save whether two samples originate from the same speaker.

    :param desc: free text description of the experiment
    :param dataset: path to transcript index file
    :param modeldir: directory for the saved 'model'
    :param resultdir: directory for saving predictions
    :param pair_file: path to file containing pre-defined pairs
    :param min_words_in_conv: min number of words a file should have for training a model
    :return: None
    """
    ds = roxsd_data.RoxsdDataSource(dataset, modeldir, min_words_in_conv=min_words_in_conv)
    X, y, conv_ids = ds.get()

    assert X.shape[0] > 0

    with open(f'{modeldir}/preprocessor.pkl', 'rb') as f:
        preprocessor1 = pickle.load(f)
    X1 = preprocessor1.transform(X)

    # with open(f'{modeldir}/preprocessor.sav', 'rb') as f:
    #     preprocessor2 = pickle.load(f)
    # X2 = preprocessor2.transform(X)
    #
    # preprocessor3 = joblib.load(f'{modeldir}/preprocessor.mod')
    # X3 = preprocessor3.transform(X)
    #
    # preprocessor4 = joblib.load(f'{modeldir}/preprocessor.joblib')
    # X4 = preprocessor4.transform(X)

    X_pairs, y_pairs, conv_pairs = get_pairs(X1, y, conv_ids, pairsdir)

    with open(f'{modeldir}/distance_classifier_calibrator.pkl', 'rb') as f:
        clf1 = pickle.load(f)

    lrs = clf1.predict_lr(X_pairs)

    cllr = lir.metrics.cllr(lrs, y_pairs)
    acc = np.mean((lrs > 1) == y_pairs)
    recall = np.mean(lrs[y_pairs == 1] > 1)  # true positive rate
    precision = np.mean(y_pairs[lrs > 1] == 1)
    tnr = np.mean(lrs[y_pairs == 0] <= 1)  # true negative rate

    fpr, tpr, threshold = roc_curve(list(y_pairs), list(lrs), pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    Metrics = collections.namedtuple('Metrics', ['cllr', 'accuracy', 'eer', 'recall', 'precision', 'tnr'])
    results = Metrics(cllr, acc, eer, recall, precision, tnr)

    res = {'metrics': results, 'lrs': lrs.tolist(), 'y_pair': y_pairs.tolist(), 'conv_pair_ids': conv_pairs.tolist()}

    with open(resultdir, 'w', encoding='utf-8') as f:
        f.write(json.dumps(res, indent=4))

    return results


def run(dataset, modeldir, resultdir, pairsdir=None):

    exp = experiments.Evaluation(predict_samesource)

    exp.parameter('dataset', dataset)
    exp.parameter('pairsdir', pairsdir)
    exp.parameter('resultdir', resultdir)
    exp.parameter('modeldir', modeldir)

    exp.parameter('min_words_in_conv', 10)

    try:
        exp.runDefaults()

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, file=sys.stdout)


if __name__ == '__main__':
    config = confidence.load_name('authorship_predict', 'local')
    warnings.filterwarnings("error")
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', help='increases verbosity', action='count', default=0)
    parser.add_argument('-q', help='decreases verbosity', action='count', default=0)
    parser.add_argument('--data', metavar='FILENAME',
                        help=f'dataset to be used; index file as generated by `sha256sum` (default: {config.data})',
                        default=config.data)
    parser.add_argument('--pairs_directory', metavar='PAIRSDIRNAME',
                        help=f'predefined pairs` (default: {config.pairsdir})',
                        default=config.pairsdir)
    parser.add_argument('--model_directory', metavar='MODELDIRNAME',
                        help=f'path to saved model files (default: {config.modeldir})', default=config.modeldir)
    parser.add_argument('--output_directory', '-o', metavar='DIRNAME',
                        help=f'path to generated output files (default: {config.resultdir})', default=config.resultdir)
    args = parser.parse_args()

    os.makedirs(args.output_directory, exist_ok=True)
    run(args.data, args.model_directory, args.output_directory, args.pairs_directory)