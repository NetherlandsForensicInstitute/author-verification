#!/usr/bin/env python3

import argparse
import collections
import logging
import os
import re
import sys
import traceback
import warnings
import confidence
import lir
import scipy.spatial
import scipy.stats
import sklearn.model_selection
import sklearn.neighbors
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import numpy as np

from functools import partial
from lir import transformers
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.metrics import roc_curve
from pyllr.bayes_error_rate import Bayes_error_rate_analysis
from scipy.special import logit

from authorship import transcriptions
from authorship import experiments

DEFAULT_LOGLEVEL = logging.WARNING
LOG = logging.getLogger(__name__)


def setupLogging(args):
    loglevel = max(logging.DEBUG, min(logging.CRITICAL, DEFAULT_LOGLEVEL + (args.q - args.v) * 10))

    # setup formatter
    log_format = '[%(asctime)-15s %(levelname)s] %(name)s: %(message)s'
    fmt = logging.Formatter(log_format)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(loglevel)
    logging.getLogger().addHandler(ch)

    # setup a file handler
    fh = logging.FileHandler('run.log', mode='w')
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)

    logging.getLogger('').setLevel(logging.DEBUG)


class KdeCdfTransformer(sklearn.base.TransformerMixin):
    def __init__(self, value_range=(None, None), resolution=1000, plot_cdf=False):
        self._value_range = value_range
        self._resolution = resolution
        self._kernels = None
        self._plot_cdf = plot_cdf

    def get_range(self, feature_values):
        lower = self._value_range[0] or np.min(feature_values)
        upper = self._value_range[1] or np.max(feature_values)

        return lower, upper

    def fit(self, X):
        assert len(X.shape) == 2

        self._kernels = []
        for i in range(X.shape[1]):
            feature_values = X[:, i]
            lower, upper = self.get_range(feature_values)

            kernel = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=.1).fit(feature_values.reshape(-1, 1))
            precomputed_values = np.arange(self._resolution + 1).reshape(-1, 1) / self._resolution * (
                    upper - lower) + lower
            density = np.exp(kernel.score_samples(precomputed_values))
            cumulative_density = np.cumsum(density)
            cumulative_density = cumulative_density / cumulative_density[-1]
            self._kernels.append(cumulative_density)

            if self._plot_cdf:
                plt.plot(precomputed_values, cumulative_density)

        if self._plot_cdf:
            plt.show()

        return self

    def transform(self, X):
        assert self._kernels is not None
        assert len(X.shape) == 2
        assert X.shape[1] == len(self._kernels)

        features = []
        for i in range(X.shape[1]):
            feature_values = X[:, i]
            lower, upper = self.get_range(feature_values)

            percentiles = self._kernels[i][((feature_values - lower) / (upper - lower) * self._resolution).astype(int)]
            features.append(percentiles)

        return np.stack(features, axis=1)


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


GaussParams = collections.namedtuple('StandardParams', ['mean0', 'std0', 'mean1', 'std1'])


class GaussianScorer(sklearn.base.BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        assert np.all(np.arange(np.max(y) + 1) == np.unique(
            y)), 'classes must be numbered 0..n and all classes must occur at least once'

        self._classes = np.unique(y)
        self._models = {}
        for cls in self._classes:
            X0 = X[y != cls]
            X1 = X[y == cls]

            mean0 = np.mean(X0, axis=0)
            std0 = np.std(X0, axis=0)
            mean1 = np.mean(X1, axis=0)
            std1 = np.std(X1, axis=0)

            # if parameters could not be estimated, assume std 1
            std0[std0 == 0] = 1
            std1[std1 == 0] = 1

            self._models[cls] = GaussParams(mean0, std0, mean1, std1)

    def predict_proba(self, X):
        return lir.util.to_probability(self.predict_lr(X))

    def predict(self, X):
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)

    def predict_lr(self, X):
        p = []
        for cls in self._classes:
            params = self._models[cls]
            p0 = scipy.stats.norm.pdf(X, params.mean0, params.std0)
            p1 = scipy.stats.norm.pdf(X, params.mean1, params.std1)
            with np.errstate(divide='ignore'):
                p.append(p1 / (p0 + p1))

        return np.prod(lir.to_odds(np.array(p)),
                       axis=2).T  # multiply the odds over categories (assume conditional independence)


class BrayDistance(sklearn.base.TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == 2

        left = X[:, :, 0]
        right = X[:, :, 1]

        return np.abs(right - left) / (np.abs(right + left) + 1)


class ShanDistanceVector(sklearn.base.TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == 2

        p = X[:, :, 0]
        q = X[:, :, 1]
        m = (p + q) / 2.0

        left = scipy.spatial.distance.rel_entr(p, m)
        right = scipy.spatial.distance.rel_entr(q, m)

        try:
            result = np.sqrt((left + right) / 2.)
        except:
            raise ValueError(
                'illegal input for ShanDistanceVector (relative entropy may be negative if input contains values > 1)')

        assert X.shape[0:2] == result.shape
        return result


class VectorDistance(sklearn.base.TransformerMixin):
    def __init__(self, dfunc):
        self._dfunc = dfunc

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == 2

        distance_by_pair = []
        for z in range(X.shape[0]):
            dist = self._dfunc(X[z, :, 0], X[z, :, 1])
            distance_by_pair.append(dist)

        return np.array(distance_by_pair).reshape(-1, 1)

    def predict_proba(self, X):
        p0 = self.transform(X)
        return np.stack([p0, 1 / p0], axis=1)


def get_pairs(X, y, ratio_limit=3):
    """
    X = conversations - from the transcriptions
    y = labels (speaker id) - from the transcriptions
    sample_size = maximum number of samples per label
    ratio_limit =

    return: transcriptions pairs, labels 0 or 1
    """
    # pair instances: same source and different source
    pairs_transformation = transformers.InstancePairing(ratio_limit=ratio_limit, seed=15)

    X_pairs, y_pairs = pairs_transformation.transform(X, y)

    return X_pairs, y_pairs


def get_batch_simple(X, y, conv_ids, repeats, ratio_limit, preprocessor):
    for i in range(repeats):
        authors = np.unique(y)
        authors_train, authors_test = sklearn.model_selection.train_test_split(authors, test_size=.1, random_state=i)

        # prep data for train
        X_subset_for_train = X[np.isin(y, authors_train), :]
        X_subset_for_train = preprocessor.fit_transform(X_subset_for_train)
        y_subset_for_train = y[np.isin(y, authors_train)]

        # prep data for test
        X_subset_for_test = X[np.isin(y, authors_test), :]
        X_subset_for_test = preprocessor.transform(X_subset_for_test)
        y_subset_for_test = y[np.isin(y, authors_test)]

        # max_n_of_pairs_per_class affects only the train set, the size of the test set depends on the number of
        # same source pairs
        X_train, y_train = get_pairs(X_subset_for_train, y_subset_for_train, ratio_limit)
        X_test, y_test = get_pairs(X_subset_for_test, y_subset_for_test, ratio_limit)

        yield X_train, y_train, X_test, y_test


def evaluate_samesource(desc, data_path, data_name, ground_truth, n_frequent_words, ratio_limit,
                        preprocessor, classifier, calibrator, extra_info=None, remove_filler_sounds=False,
                        expand_contractions=False, repeats=1, min_num_of_words=1):
    """
    Run an experiment with the parameters provided.

    :param desc: free text description of the experiment
    :param data_path: path to transcript index file
    :param data_name
    :param ground_truth
    :param n_frequent_words: int: number of most frequent words to be used in the analysis
    :param ratio_limit: ratio of different-source to same-source pairs to be considered
    :param preprocessor: a sklearn pipeline
    :param classifier: a sklearn pipeline with a classifier as last element
    :param calibrator: a LIR calibrator
    :param extra_info
    :param remove_filler_sounds
    :param expand_contractions
    :param repeats: int: the number of times the experiment is run
    :param min_num_of_words

    :return: cllr, accuracy, eer, recall, precision (if all_metrics=True then cllr, cllr_min, cllr_cal, accuracy, eer,
             recall, true negative rate, precision, mean_logLR_diff, mean_logLR_same)
    """

    clf = lir.CalibratedScorer(classifier, calibrator)  # set up classifier and calibrator for authorship technique

    # load data
    ds = transcriptions.DataSource(data_path=data_path, data_name=data_name, ground_truth=ground_truth,
                                   extra_info=extra_info, remove_filler_sounds=remove_filler_sounds,
                                   expand_contractions=expand_contractions, n_frequent_words=n_frequent_words,
                                   min_num_of_words=min_num_of_words)
    X, y, conv_ids = ds.get()

    assert X.shape[0] > 0

    desc_pre = '; '.join(name for name, tr in preprocessor.steps)
    desc_clf = '; '.join(name for name, tr in clf.scorer.steps)
    title = f'{desc}: using common source model: {desc_pre}; {desc_clf}; {ds}; repeats={repeats}'
    LOG.info(title)
    LOG.info(f'{desc}: number of speakers: {np.unique(y).size}')
    LOG.info(f'{desc}: number of files: {y.size}')

    lrs_mfw = []
    y_all = []

    for X_train, y_train, X_test, y_test in tqdm(get_batch_simple(X, y, conv_ids, repeats, ratio_limit,
                                                                  preprocessor)):

        clf.fit(X_train, y_train)
        lrs_mfw.append(clf.predict_lr(X_test))
        y_all.append(y_test)

        n_same_train = int(np.sum(y_train))
        n_diff_train = int(y_train.size - n_same_train)
        n_same = int(np.sum(y_test))
        n_diff = int(y_test.size - n_same)

        LOG.info(f'  counts by class (train): diff={n_diff_train}; same={n_same_train}')
        LOG.info(f'  counts by class: diff={n_diff}; same={n_same}')

    # calculate metrics and log them
    mfw_res, stds, eers = calculate_metrics(lrs_mfw, y_all)

    LOG.info(f'  mfw only: {mfw_res._fields} = {list(np.round(mfw_res, 3))}')

    return [mfw_res, stds, eers]


def calculate_metrics(lrs, y):
    """
    calculate several metrics

    :param lrs: lrs as derived by the evaluate_samesource
    :param y: true labels

    :return: [namedtuple] eer, minDCF(at 0.0, 0.005, 0.01, 0.1, 0.5)
    """
    eers = []
    pavs = []

    p = [0.005, 0.01, 0.1, 0.5]
    for i in range(len(lrs)):
        fpr, tpr, threshold = roc_curve(list(y[i]), list(lrs[i]), pos_label=1)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        eers.append(eer)

        pav, _, _ = Bayes_error_rate_analysis(lrs[i], y[i], logit(p))
        pavs.append(pav)

    eer_avg = np.mean(eers)
    eer_std = np.std(eers)
    pav_avg = np.mean(pavs, axis=0)
    pav_std = np.std(pavs, axis=0)

    names = ['eer'] + ['minDFC_' + str(i).replace('.', '_') for i in reversed(p)]
    Metrics = collections.namedtuple(typename='Metrics', field_names=names)
    res = Metrics(eer_avg, pav_avg[3], pav_avg[2], pav_avg[1], pav_avg[0])
    stds = Metrics(eer_std, pav_std[3], pav_std[2], pav_std[1], pav_std[0])

    return res, stds, eers


def aggregate_results(out_dir, results):
    '''
    prints resutls

    :param out_dir: [outdated at the moment]
    :param results: namedtuple (as return by the evaluate_samesource)

    '''
    res_file = open(out_dir, 'w')
    for params, result in results:
        desc = ', '.join(f'{name}={value}' for name, value in params)
        print(f'mfw: {result[0]._fields} = {list(np.round(result[0], 3))}--{desc}')
        res_file.write(f'mfw_avg: {result[0]._fields} = {list(np.round(result[0], 3))}--{desc}\n'
                       f'mfw_std: {result[1]._fields} = {list(np.round(result[1], 3))}--{desc}\n')
    res_file.close()


def output_file_name(data_path, data_name, ground_truth, exclude_fillers, expand_contractions, n_freq_words,
                     output_directory):
    path_safe = re.sub('[^a-zA-Z0-9_-]', '_', os.path.basename(data_path))

    if data_name in path_safe:
        base_name = path_safe
    else:
        base_name = data_name + '_' + path_safe

    filename = base_name + '_gt_' + str(ground_truth)[0]
    if exclude_fillers:
        filename = filename + '_nofillers'
    else:
        filename = filename + '_withfillers'

    if expand_contractions:
        filename = filename + '_expand_contr'

    filename = filename + '_' + str(n_freq_words) + '_mfw.txt'

    return os.path.join(output_directory, filename)


def run(data_path, data_name, ground_truth, extra_file, exclude_fillers, expand_contractions, n_freq_words, resultdir):

    # PREPROCESSORS for authorship verification
    prep_none = sklearn.pipeline.Pipeline([
        ('scale:none', None),
        ('pop:none', None),
    ])

    prep_std = sklearn.pipeline.Pipeline([
        ('scale:standard', sklearn.preprocessing.StandardScaler()),
        ('pop:none', None),
    ])

    prep_norm = sklearn.pipeline.Pipeline([
        ('scale:norm', sklearn.preprocessing.Normalizer()),
        ('pop:none', None),
    ])

    prep_gauss = sklearn.pipeline.Pipeline([
        # ('scale:standard', sklearn.preprocessing.StandardScaler()),
        ('pop:gauss', GaussianCdfTransformer()),  # cumulative density function for each feature
        # ('pop:gauss', sklearn.preprocessing.QuantileTransformer()),  # cumulative density function for each feature
    ])

    prep_kde = sklearn.pipeline.Pipeline([
        ('scale:standard', sklearn.preprocessing.StandardScaler()),
        ('pop:kde', KdeCdfTransformer()),  # cumulative density function for each feature
    ])

    # CLASSIFIERS for authorship verification (an element-wise distance and a binary ML alg is expected)
    logit = sklearn.pipeline.Pipeline([
        ('diff:abs', transformers.AbsDiffTransformer()),
        ('clf:logit', LogisticRegression(max_iter=600, class_weight='balanced')),
    ])

    logit_br = sklearn.pipeline.Pipeline([
        ('bray', BrayDistance()),
        ('clf:logit', LogisticRegression(max_iter=600, class_weight='balanced')),
    ])

    svm = sklearn.pipeline.Pipeline([
        ('diff:abs', transformers.AbsDiffTransformer()),
        # ('diff:shan', ShanDistanceVector()),
        ('clf:svc', sklearn.svm.SVC(gamma='scale', kernel='linear', probability=True, class_weight='balanced')),
    ])

    svm_br = sklearn.pipeline.Pipeline([
        ('diff:bray', BrayDistance()),
        ('clf:svc', sklearn.svm.SVC(gamma='scale', kernel='linear', probability=True, class_weight='balanced')),
    ])

    exp = experiments.Evaluation(evaluate_samesource, partial(aggregate_results, resultdir))

    exp.parameter('data_path', data_path)
    exp.parameter('data_name', data_name)
    exp.parameter('ground_truth', ground_truth)
    exp.parameter('extra_info', extra_file)
    exp.parameter('remove_filler_sounds', exclude_fillers)
    exp.parameter('expand_contractions', expand_contractions)

    exp.parameter('min_num_of_words', 20)

    exp.parameter('n_frequent_words', n_freq_words)
    exp.addSearch('n_frequent_words', [args.n_freq_words-100, n_freq_words+100,
                                       n_freq_words+200], include_default=False)

    exp.parameter('ratio_limit', 3)

    exp.parameter('preprocessor', prep_gauss)

    exp.parameter('classifier', ('bray_logit', logit_br))
    exp.addSearch('classifier', [('br_mlp', svm_br), ('bray_logit', logit_br)],
                  include_default=False)

    # exp.parameter('calibrator', lir.ELUBbounder(lir.LogitCalibrator()))
    exp.parameter('calibrator', lir.DummyLogOddsCalibrator())

    exp.parameter('repeats', 100)

    try:
        exp.runDefaults()
        # exp.runSearch('max_n_of_pairs_per_class')
        # exp.runFullGrid(['n_frequent_words', 'max_n_of_pairs_per_class'])

    except Exception as e:
        LOG.fatal(e.args[1])
        LOG.fatal(e.args[0])
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, file=sys.stdout)


if __name__ == '__main__':
    config = confidence.load_name('authorship', 'local')
    warnings.filterwarnings("error")
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', help='increases verbosity', action='count', default=0)
    parser.add_argument('-q', help='decreases verbosity', action='count', default=0)
    parser.add_argument('--data_path', metavar='DIRNAME', default=config.data.path)
    parser.add_argument('--data_name', type=str, default=config.data.name)
    parser.add_argument('--ground_truth', default=config.data.ground_truth)
    parser.add_argument('--extra_file', default=config.data.extra_info)
    parser.add_argument('--exclude_fillers', default=config.data_process.exclude_fillers)
    parser.add_argument('--expand_contractions', default=config.data_process.expand_contractions)
    parser.add_argument('--n_freq_words', default=config.data_process.n_frequent_words)
    parser.add_argument('--output_directory', default=config.resultdir)
    args = parser.parse_args()

    setupLogging(args)

    os.makedirs(args.output_directory, exist_ok=True)
    output_file = output_file_name(args.data_path, args.data_name, args.ground_truth, args.exclude_fillers,
                                   args.expand_contractions, args.n_freq_words, args.output_directory)

    run(args.data_path, args.data_name, args.ground_truth, args.extra_file, args.exclude_fillers,
        args.expand_contractions, args.n_freq_words, output_file)
