#!/usr/bin/env python3

import argparse
import collections
import logging
import os
import sys
import traceback
import warnings
from functools import partial

import confidence
import lir
from lir import transformers
from matplotlib import pyplot as plt
import numpy as np
import json
import scipy.spatial
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import sklearn.model_selection
import sklearn.neighbors
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score

from authorship import data
from authorship import vocalize_data
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


class makeplots:
    def __init__(self, path_prefix=None):
        self.path_prefix = path_prefix

    def __call__(self, lrs, y, title='', shortname=''):
        n_same = int(np.sum(y))
        n_diff = int(y.size - np.sum(y))
        cllr = lir.metrics.cllr(lrs, y)
        acc = np.mean((lrs > 1) == y)

        LOG.info(f'  total counts by class (sum of all repeats): diff={n_diff}; same={n_same}')
        LOG.info(
            f'  average LR by class: 1/{np.exp(np.mean(-np.log(lrs[y == 0])))}; {np.exp(np.mean(np.log(lrs[y == 1])))}')
        LOG.info(
            f'  cllr, acc: {cllr, acc}')

        path_prefix = os.path.join(self.path_prefix, shortname.replace('*', ''))
        tippet_path = f'{path_prefix}_tippet.png' if self.path_prefix is not None else None
        pav_path = f'{path_prefix}_pav.png' if self.path_prefix is not None else None
        ece_path = f'{path_prefix}_ece.png' if self.path_prefix is not None else None

        kw_figure = {}

        lir.plotting.plot_tippett(lrs, y, savefig=tippet_path, kw_figure=kw_figure)
        lir.plotting.plot_pav(lrs, y, savefig=pav_path, kw_figure=kw_figure)
        lir.ece.plot(lrs, y, path=ece_path, on_screen=not ece_path, kw_figure=kw_figure)


def get_pairs(X, y, conv_ids, voc_conv_pairs, voc_scores, sample_size):
    """
    X = conversations - from the transcriptions
    y = labels (speaker id) - from the transcriptions
    conv_ids = conversation id per instance - from the transcriptions
    voc_conv_pairs = pair of conversation ids compared using vocalise
    voc_scores = vocalise score for each pair in the voc_conv_pairs
    sample_size = maximum number of samples per label

    actual number of samples is expected to be smaller if vocalize scores are missing!!

    return: transcriptions pairs, labels 0 or 1, vocalize score
    """
    # pair instances: same source and different source
    pairs_transformation = transformers.InstancePairing(same_source_limit=int(sample_size),
                                                        different_source_limit=int(sample_size))
    X_pairs, y_pairs = pairs_transformation.transform(X, y)
    pairing = pairs_transformation.pairing  # indices of pairs based on the transcriptions
    conv_pairs = np.apply_along_axis(lambda a: np.array([conv_ids[a[0]], conv_ids[a[1]]]), 1, pairing)  # from indices to the actual pairs

    # search the pair based on transcription within the conv_ids (order of the speaker ids is not crucial)
    # and return the vocalise score if no score exists set value to NaN
    voc_scores_subset = np.apply_along_axis(
        lambda a: voc_scores[np.where(voc_conv_pairs == set(a))[0]][0][0]
        if len(np.where(voc_conv_pairs == set(a))[0]) == 1 else np.NaN, 1, conv_pairs)

    # to be able to combine the two systems we work only with the data that overlap
    voc_score_clean = voc_scores_subset[~np.isnan(voc_scores_subset)]
    y_pairs_clean = y_pairs[~np.isnan(voc_scores_subset)]
    X_pairs_clean = X_pairs[~np.isnan(voc_scores_subset), :, :]

    return X_pairs_clean, voc_score_clean, y_pairs_clean


def get_batch_simple(X, y, conv_ids, voc_conv_pairs, voc_scores, repeats, max_n_of_pairs_per_class, preprocessor):
    for i in range(repeats):
        authors = np.unique(y)
        authors_train, authors_test = sklearn.model_selection.train_test_split(authors, test_size=.1, random_state=i)

        # prep data for train
        X_subset_for_train = X[np.isin(y, authors_train), :]
        X_subset_for_train = preprocessor.fit_transform(X_subset_for_train)
        y_subset_for_train = y[np.isin(y, authors_train)]
        conv_ids_subset_for_train = conv_ids[np.isin(y, authors_train)]

        # prep data for test
        X_subset_for_test = X[np.isin(y, authors_test), :]
        X_subset_for_test = preprocessor.transform(X_subset_for_test)
        y_subset_for_test = y[np.isin(y, authors_test)]
        conv_ids_subset_for_test = conv_ids[np.isin(y, authors_test)]

        X_train, X_voc_train, y_train = get_pairs(X_subset_for_train, y_subset_for_train, conv_ids_subset_for_train,
                                                  voc_conv_pairs, voc_scores, max_n_of_pairs_per_class)
        X_test, X_voc_test, y_test = get_pairs(X_subset_for_test, y_subset_for_test, conv_ids_subset_for_test,
                                               voc_conv_pairs, voc_scores, 2000)

        yield X_train, X_voc_train, y_train, X_test, X_voc_test, y_test


def evaluate_samesource(desc, dataset, voc_data, device, n_frequent_words, max_n_of_pairs_per_class, preprocessor, classifier,
                        calibrator, plot=None, repeats=1, min_num_of_words=0, all_metrics=False):
    """
    Run an experiment with the parameters provided.

    :param desc: free text description of the experiment
    :param dataset: path to transcript index file
    :param voc_data: path to vocalise output
    :param n_frequent_words: int: number of most frequent words to be used in the analysis
    :param max_n_of_pairs_per_class: maximum number of pairs per class (same- or different-source) to be taken
    :param preprocessor: Pipeline: an sklearn pipeline
    :param classifier: Pipeline: an sklearn pipeline with a classifier as last element
    :param calibrator: a LIR calibrator
    :param plot: a plotting function
    :param repeats: int: the number of times the experiment is run

    :return: cllr, accuracy, eer, recall, precision (if all_metrics=True then cllr, cllr_min, cllr_cal, accuracy, eer,
             recall, true negative rate, precision, mean_logLR_diff, mean_logLR_same)
    """

    clf = lir.CalibratedScorer(classifier, calibrator)  # set up classifier and calibrator for authorship technique
    voc_cal = lir.ScalingCalibrator(lir.LogitCalibrator())  # set up calibrator for vocalise output
    mfw_voc_clf = lir.CalibratedScorer(LogisticRegression(class_weight='balanced'), calibrator)  # set up logit as classifier and calibrator for a type of fusion
    combine_features_flag = False  # In current setting, a distance scorer always has 1 step while a ML has 2
    if len(clf.scorer.named_steps) > 1:
        combine_features_flag = True
        features_clf = lir.CalibratedScorer(clf.scorer.steps[1][1], calibrator)  # set up classifier and calibrator for a type of fusion

    ds = data.DataSource(dataset, n_frequent_words=n_frequent_words, min_num_of_words=min_num_of_words)
    X, y, conv_ids = ds.get()
    voc_ds = vocalize_data.VocalizeDataSource(voc_data, device=device)
    voc_conv_pairs, voc_scores = voc_ds.get()

    assert X.shape[0] > 0

    desc_pre = '; '.join(name for name, tr in preprocessor.steps)
    desc_clf = '; '.join(name for name, tr in clf.scorer.steps)
    title = f'{desc}: using common source model: {desc_pre}; {desc_clf}; {ds}; repeats={repeats}'
    LOG.info(title)
    LOG.info(f'{desc}: number of speakers: {np.unique(y).size}')
    LOG.info(f'{desc}: number of instances: {y.size}')

    # X = preprocessor.fit_transform(X)  # shouldn't this take place on the x_train and then on x_test?

    lrs_mfw = []
    lrs_voc = []
    lrs_comb_a = []
    lrs_comb_b = []
    lrs_features = []
    y_all = []

    for X_train, X_voc_train, y_train, X_test, X_voc_test, y_test in tqdm(
            get_batch_simple(X, y, conv_ids, voc_conv_pairs, voc_scores, repeats,
                             max_n_of_pairs_per_class, preprocessor)):

        # preprocessing the data - NOT WORKING - not the place to be, at t
        # X_train = preprocessor.fit_transform(X_train)
        # X_test = preprocessor.transform(X_test)

        # there are three ways to combine the mfw method with the voc output
        # 1. assume that mfw and voc LR are independent and multiply them (m1)
        # 2a. apply logit using as input the mfc and the voc score, then calibrate the resulted score (m2a)
        # 2b. apply logit using as input the mfc score, the voc score, and their product, then calibrate
        #     the resulted score (m2a)
        # 3. use the voc score as additional feature to the mfw input vector (m3)
        # NOTE: m3 can be used only if the scorer is a classification alg, otherwise m3 = m2 with scorer always a logit

        # calculate LRs for vocalize output (for m1)
        voc_cal.fit(X=X_voc_train, y=y_train)
        lrs_voc.append(voc_cal.transform(X_voc_test))

        # mfw - fit a classifier and calculate LRs (for m1, the resulted scorer can also be used for m2)
        clf.fit(X_train, y_train)
        lrs_mfw.append(clf.predict_lr(X_test))
        y_all.append(y_test)

        # take scores from mfw scorer and combine with voc output using logit then calibrate (for m2)
        mfw_scores_train = lir.apply_scorer(clf.scorer, X_train)

        # scale voc_score to match the value range of the mfw clasifier (0-1)
        scaler = sklearn.preprocessing.MinMaxScaler()
        scaler.fit(X_voc_train.reshape(-1, 1))
        X_voc_train_norm = scaler.transform(X_voc_train.reshape(-1, 1)).T
        X_voc_test_norm = scaler.transform(X_voc_test.reshape(-1, 1)).T

        # X_comb_train = np.vstack((np.squeeze(mfw_scores_train), X_voc_train_norm)).T  # don't recall why squeeze
        X_comb_train_a = np.vstack((mfw_scores_train, X_voc_train_norm)).T
        mfw_voc_clf.fit(X_comb_train_a, y_train)

        mfw_scores_test = lir.apply_scorer(clf.scorer, X_test)
        # X_comb_test = np.vstack((np.squeeze(mfw_scores_test), X_voc_test_norm)).T
        X_comb_test_a = np.vstack((mfw_scores_test, X_voc_test_norm)).T
        lrs_comb_a.append(mfw_voc_clf.predict_lr(X_comb_test_a))

        prod_train = X_comb_train_a[:, 0]*X_comb_train_a[:, 1]
        X_comb_train_b = np.vstack((mfw_scores_train, X_voc_train_norm, prod_train)).T
        mfw_voc_clf.fit(X_comb_train_b, y_train)
        prod_test = X_comb_test_a[:, 0] * X_comb_test_a[:, 1]
        X_comb_test_b = np.vstack((mfw_scores_test, X_voc_test_norm, prod_test)).T
        lrs_comb_b.append(mfw_voc_clf.predict_lr(X_comb_test_b))

        # check type of scorer (for m3). In current setting, a distance scorer always has 1 step while a ML has 2
        if combine_features_flag:
            X_train_onevector = clf.scorer.steps[0][1].transform(X_train)
            X_train_onevector = np.column_stack((X_train_onevector, X_voc_train_norm.T))
            features_clf.fit(X_train_onevector, y_train)

            X_test_onevector = clf.scorer.steps[0][1].transform(X_test)
            X_test_onevector = np.column_stack((X_test_onevector, X_voc_test_norm.T))
            lrs_features.append(features_clf.predict_lr(X_test_onevector))

    lrs_mfw = np.concatenate(lrs_mfw)
    lrs_voc = np.concatenate(lrs_voc)
    lrs_comb_a = np.concatenate(lrs_comb_a)
    lrs_comb_b = np.concatenate(lrs_comb_b)
    if combine_features_flag:
        lrs_features = np.concatenate(lrs_features)
    y_all = np.concatenate(y_all)

    if plot is not None:
        plot(lrs_mfw, y_all, title=title, shortname=desc)

    mfw_res = calculate_metrics(lrs_mfw, y_all, full_list=all_metrics)
    voc_res = calculate_metrics(lrs_voc, y_all, full_list=all_metrics)
    lrs_multi = np.multiply(lrs_mfw, lrs_voc)
    lrs_multi_res = calculate_metrics(lrs_multi, y_all)
    comb_res_a = calculate_metrics(lrs_comb_a, y_all, full_list=all_metrics)
    comb_res_b = calculate_metrics(lrs_comb_b, y_all, full_list=all_metrics)
    if combine_features_flag:
        feat_res = calculate_metrics(lrs_features, y_all, full_list=all_metrics)
        return mfw_res, voc_res, lrs_multi_res, comb_res_a, comb_res_b, feat_res
    else:
        return mfw_res, voc_res, lrs_multi_res, comb_res_a, comb_res_b


def calculate_metrics(lrs, y, full_list=False):

    cllr = lir.metrics.cllr(lrs, y)
    acc = np.mean((lrs > 1) == y)
    recall = np.mean(lrs[y == 1] > 1)  # true positive rate
    precision = np.mean(y[lrs > 1] == 1)

    fpr, tpr, threshold = roc_curve(list(y), list(lrs), pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    auc = roc_auc_score(list(y), list(lrs))

    if full_list:
        cllrmin = lir.metrics.cllr_min(lrs, y)
        cllrcal = cllr - cllrmin
        tnr = np.mean(lrs[y == 0] < 1)  # true negative rate
        mean_logLR_diff = np.mean(np.log(lrs[y == 0]))
        mean_logLR_same = np.mean(np.log(lrs[y == 1]))
        Metrics = collections.namedtuple('Metrics', ['cllr', 'cllrmin', 'cllrcal', 'accuracy', 'eer', 'auc', 'recall',
                                                     'tnr', 'precision', 'mean_logLR_diff', 'mean_logLR_same'])
        res = Metrics(cllr, cllrmin, cllrcal, acc, eer, auc, recall, tnr, precision, mean_logLR_diff, mean_logLR_same)
    else:
        Metrics = collections.namedtuple('Metrics', ['cllr', 'accuracy', 'eer', 'auc', 'recall', 'precision'])
        res = Metrics(cllr, acc, eer, auc, recall, precision)

    return res


def aggregate_results(out_dir, results):
    for params, result in results:
        desc = ', '.join(f'{name}={value}' for name, value in params)
        print(f'mfw only: {result[0]._fields} = {list(np.round(result[0], 3))}--{desc}')
        print(f'voc only: {result[1]._fields} = {list(np.round(result[1], 3))}--{desc}')
        print(f'lrs comb by multi: {result[2]._fields} = {list(np.round(result[2], 3))}--{desc}')
        print(f'lrs comb by logiA: {result[3]._fields} = {list(np.round(result[3], 3))}--{desc}')
        print(f'lrs comb by logiB: {result[4]._fields} = {list(np.round(result[4], 3))}--{desc}')
        if len(result) == 6:
            print(f'lrs comb by featu: {result[5]._fields} = {list(np.round(result[5], 3))}--{desc}')


def run(dataset, voc_data, resultdir):
    ### PREPROCESSORS for authorship verification

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
        ('pop:gauss', GaussianCdfTransformer()),  # cumulative density function for each feature
        # ('pop:gauss', sklearn.preprocessing.QuantileTransformer()),  # cumulative density function for each feature
    ])

    prep_kde = sklearn.pipeline.Pipeline([
        ('scale:standard', sklearn.preprocessing.StandardScaler()),
        ('pop:kde', KdeCdfTransformer()),  # cumulative density function for each feature
    ])

    ### CLASSIFIERS for authorship verification

    dist_br = sklearn.pipeline.Pipeline([
        ('dist:bray', VectorDistance(scipy.spatial.distance.braycurtis)),
    ])

    dist_ma = sklearn.pipeline.Pipeline([
        ('dist:man', VectorDistance(scipy.spatial.distance.cityblock)),
    ])

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
        # ('diff:bray', BrayDistance()),
        ('clf:svc', sklearn.svm.SVC(gamma='scale', kernel='linear', probability=True, class_weight='balanced')),
    ])

    svm_br = sklearn.pipeline.Pipeline([
        ('diff:bray', BrayDistance()),
        ('clf:svc', sklearn.svm.SVC(gamma='scale', kernel='linear', probability=True, class_weight='balanced')),
    ])

    br_mlp = sklearn.pipeline.Pipeline([
        ('diff:bray', BrayDistance()),
        ('clf:mlp', MLPClassifier(solver='adam', max_iter=800, alpha=0.001, hidden_layer_sizes=(5, ), random_state=1)),
    ])

    exp = experiments.Evaluation(evaluate_samesource, partial(aggregate_results, resultdir))

    exp.parameter('dataset', dataset)
    exp.parameter('voc_data', voc_data)
    exp.parameter('device', 'telephone')  # options: telephone, headset, SM58close, AKGC400BL, SM58far
    # exp.parameter('plot', makeplots(resultdir))

    exp.parameter('min_num_of_words', 0)

    exp.parameter('n_frequent_words', 200)
    exp.addSearch('n_frequent_words', [100, 200, 300], include_default=False)

    exp.parameter('max_n_of_pairs_per_class', 2500)
    exp.addSearch('max_n_of_pairs_per_class', [3000, 4000], include_default=False)

    exp.parameter('preprocessor', prep_gauss)

    exp.parameter('classifier', ('bray_logit', logit_br))
    exp.addSearch('classifier', [('man_logit', logit), ('dist_man', dist_ma), ('bray_svm', svm_br), ('bray_logit', logit_br)], include_default=False)

    exp.parameter('calibrator', lir.ScalingCalibrator(lir.KDECalibrator()))
    exp.parameter('repeats', 10)

    try:
        # exp.runDefaults()
        exp.runSearch('max_n_of_pairs_per_class')
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
    parser.add_argument('--data', metavar='FILENAME',
                        help=f'dataset to be used; index file as generated by `sha256sum` (default: {config.data})',
                        default=config.data)
    parser.add_argument('--vocalise-data', metavar='FILENAME',
                        help=f'vocalize output to be used; (default: {config.vocalise_data})',
                        default=config.vocalise_data)
    parser.add_argument('--output-directory', '-o', metavar='DIRNAME',
                        help=f'path to generated output files (default: {config.resultdir})', default=config.resultdir)
    args = parser.parse_args()

    setupLogging(args)

    os.makedirs(args.output_directory, exist_ok=True)
    run(args.data, args.vocalise_data, args.output_directory)
