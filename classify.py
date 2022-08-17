#!/usr/bin/env python3

import argparse
import collections
import logging
import os
import sys
import traceback
import warnings
import json
from functools import partial

import confidence
import lir
from lir import transformers
from matplotlib import pyplot as plt
import numpy as np
import scipy.spatial
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import sklearn.model_selection
import sklearn.neighbors
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score

from authorship import frida_data
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


def get_pairs(X, y, conv_ids, voc_conv_pairs, voc_scores, sample_size):
    """
    X = conversations - from the transcriptions
    y = labels (speaker id) - from the transcriptions
    conv_ids = conversation id per instance - from the transcriptions
    voc_conv_pairs = pair of conversation ids compared using vocalise
    voc_scores = vocalise score for each pair in the voc_conv_pairs
    sample_size = maximum number of samples per label

    actual number of samples is expected to be smaller if vocalise scores are missing!!

    return: transcriptions pairs, labels 0 or 1, vocalise score
    """
    # pair instances: same source and different source
    pairs_transformation = transformers.InstancePairing(same_source_limit=int(sample_size),
                                                        different_source_limit=int(sample_size),
                                                        seed=15)
    X_pairs, y_pairs = pairs_transformation.transform(X, y)
    pairing = pairs_transformation.pairing  # indices of pairs based on the transcriptions
    conv_pairs = np.apply_along_axis(lambda a: np.sort(np.array([conv_ids[a[0]], conv_ids[a[1]]])), 1,
                                     pairing)  # from indices to the actual pairs
    conv_pairs = np.apply_along_axis(lambda a: str(a[0]+a[1]), 1, conv_pairs)

    # alignment of voc score to pairs for mfw
    voc_dict = dict({})
    for A, B in zip(voc_conv_pairs, voc_scores):
        voc_dict[str(A)] = B.item()

    voc_scores_subset = []
    for a in conv_pairs:
        voc_scores_subset.append(voc_dict.get(str(a), np.NaN))
    voc_scores_subset = np.array(voc_scores_subset)

    # to be able to combine the two systems we work only with the data that overlap
    voc_score_clean = voc_scores_subset[~np.isnan(voc_scores_subset)]
    y_pairs_clean = y_pairs[~np.isnan(voc_scores_subset)]
    X_pairs_clean = X_pairs[~np.isnan(voc_scores_subset), :, :]
    conv_pairs_clean = conv_pairs[~np.isnan(voc_scores_subset)]

    return X_pairs_clean, voc_score_clean, y_pairs_clean, conv_pairs_clean


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

        # max_n_of_pairs_per_class affects only the train set, the size of the test set is fixed (for fair comparisons
        # of runs with different max_n_of_pairs_per_class).
        X_train, X_voc_train, y_train, conv_pairs_train = get_pairs(X_subset_for_train, y_subset_for_train,
                                                                    conv_ids_subset_for_train, voc_conv_pairs,
                                                                    voc_scores, max_n_of_pairs_per_class)
        X_test, X_voc_test, y_test, conv_pairs_test = get_pairs(X_subset_for_test, y_subset_for_test,
                                                                conv_ids_subset_for_test, voc_conv_pairs, voc_scores,
                                                                2000)

        yield X_train, X_voc_train, y_train, X_test, X_voc_test, y_test, conv_pairs_test


def evaluate_samesource(desc, dataset, voc_data, device, n_frequent_words, max_n_of_pairs_per_class, preprocessor,
                        classifier, calibrator, repeats=1, min_num_of_words=0, all_metrics=False):
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
    :param repeats: int: the number of times the experiment is run

    :return: cllr, accuracy, eer, recall, precision (if all_metrics=True then cllr, cllr_min, cllr_cal, accuracy, eer,
             recall, true negative rate, precision, mean_logLR_diff, mean_logLR_same)
    """

    clf = lir.CalibratedScorer(classifier, calibrator)  # set up classifier and calibrator for authorship technique
    voc_cal = lir.LogitCalibratorInProbabilityDomain()  # set up calibrator for vocalise output
    # mfw_voc_clf = lir.CalibratedScorer(LogisticRegression(class_weight='balanced'),
    #                                    calibrator)  # set up logit as classifier and calibrator for a type of fusion
    mfw_voc_clf = lir.CalibratedScorer(SVC(gamma='scale', kernel='linear', probability=True, class_weight='balanced'),
                                       calibrator)
    features_clf = lir.CalibratedScorer(clf.scorer.steps[1][1],
                                        calibrator)  # set up classifier and calibrator for a type of fusion
    biva_cal = lir.LogitCalibratorInProbabilityDomain()  # set up calibrator for a type of fusion

    ds = frida_data.FridaDataSource(dataset, n_frequent_words=n_frequent_words, min_num_of_words=min_num_of_words)
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

    lrs_mfw = []
    lrs_voc = []
    lrs_comb_a = []
    lrs_comb_b = []
    lrs_features = []
    lrs_biva = []
    y_all = []
    results_to_save = collections.defaultdict(dict)

    for count, (X_train, X_voc_train, y_train, X_test, X_voc_test, y_test, conv_pairs_test) in tqdm(enumerate(
            get_batch_simple(X, y, conv_ids, voc_conv_pairs, voc_scores, repeats,
                             max_n_of_pairs_per_class, preprocessor))):

        # keep pairs and flags for test per repeat
        results_to_save[count]['train'] = y_train.tolist()
        results_to_save[count]['pairs'] = conv_pairs_test.tolist()
        results_to_save[count]['y'] = y_test.tolist()

        # the following ways are considered for combining the mfw method with the voc output
        # 1. assume that mfw and voc LR are independent and multiply them (m1)
        # 2a. apply svm using as input the mfc and the voc score, then calibrate the resulted score (m2a)
        # 2b. apply svm using as input the mfc score, the voc score, and their product, then calibrate
        #     the resulted score (m2a)
        # 3. use the voc score as additional feature to the mfw input vector (m3)
        # 4. per class, fit bivariate normal distribution on the mfw scorers and voc output (m4)

        # calculate LRs for vocalize output (for m1)
        # remember: vocalize outputs uncalibrated LRs
        voc_cal.fit(X=X_voc_train, y=y_train)
        lrs_voc.append(voc_cal.transform(X_voc_test))

        # mfw - fit classifier and calculate LRs (for m1, the resulted scorer can also be used for both variants of m2)
        clf.fit(X_train, y_train)
        lrs_mfw.append(clf.predict_lr(X_test))
        y_all.append(y_test)

        n_same = int(np.sum(y_test))
        n_diff = int(y.size - n_same)

        LOG.info(f'  counts by class: diff={n_diff}; same={n_same}')

        # scale voc_score to match the value range of the mfw classifier/data prep for mfw (0-1) (for m2 and m3)
        scaler = sklearn.preprocessing.MinMaxScaler()
        scaler.fit(X_voc_train.reshape(-1, 1))
        X_voc_train_norm = scaler.transform(X_voc_train.reshape(-1, 1)).T
        X_voc_test_norm = scaler.transform(X_voc_test.reshape(-1, 1)).T

        # take scores from mfw scorer
        mfw_scores_train = lir.apply_scorer(clf.scorer, X_train)

        # ... and combine with voc output using logit then calibrate (for m2a)
        X_comb_train_a = np.vstack((mfw_scores_train, X_voc_train_norm)).T
        mfw_voc_clf.fit(X_comb_train_a, y_train)

        mfw_scores_test = lir.apply_scorer(clf.scorer, X_test)
        X_comb_test_a = np.vstack((mfw_scores_test, X_voc_test_norm)).T
        lrs_comb_a.append(mfw_voc_clf.predict_lr(X_comb_test_a))

        # ... and combine with voc output and their product using logit then calibrate (for m2b)
        prod_train = X_comb_train_a[:, 0] * X_comb_train_a[:, 1]
        X_comb_train_b = np.vstack((mfw_scores_train, X_voc_train_norm, prod_train)).T
        mfw_voc_clf.fit(X_comb_train_b, y_train)
        prod_test = X_comb_test_a[:, 0] * X_comb_test_a[:, 1]
        X_comb_test_b = np.vstack((mfw_scores_test, X_voc_test_norm, prod_test)).T
        lrs_comb_b.append(mfw_voc_clf.predict_lr(X_comb_test_b))

        # append voc output to mfw features then fit scorer and calibrate (for m3)
        X_train_onevector = clf.scorer.steps[0][1].transform(X_train)
        X_train_onevector = np.column_stack((X_train_onevector, X_voc_train_norm.T))
        features_clf.fit(X_train_onevector, y_train)

        X_test_onevector = clf.scorer.steps[0][1].transform(X_test)
        X_test_onevector = np.column_stack((X_test_onevector, X_voc_test_norm.T))
        lrs_features.append(features_clf.predict_lr(X_test_onevector))

        # per class, fit bivariate normal distribution on the mfw scorers and voc output then calibrate (for m4)
        # mfw score to log odds to match the voc scores
        X_comb_train_m4 = np.vstack((lir.util.to_log_odds(mfw_scores_train), X_voc_train)).T
        X_comb_test_m4 = np.vstack((lir.util.to_log_odds(mfw_scores_test), X_voc_test)).T

        # parameters of bivariate normal distribution for same source
        mean_same = np.mean(X_comb_train_m4[y_train == 1], axis=0)
        cov_same = np.cov(X_comb_train_m4[y_train == 1], rowvar=0)

        # parameters of bivariate normal distribution for diff source
        mean_diff = np.mean(X_comb_train_m4[y_train == 0], axis=0)
        cov_diff = np.cov(X_comb_train_m4[y_train == 0], rowvar=0)

        # calculate the uncallibrated lrs for train and test
        scores_same = scipy.stats.multivariate_normal.pdf(X_comb_train_m4, mean=mean_same, cov=cov_same)
        scores_diff = scipy.stats.multivariate_normal.pdf(X_comb_train_m4, mean=mean_diff, cov=cov_diff)
        uncal_lr_train = np.divide(scores_same, scores_diff)

        scores_same_test = scipy.stats.multivariate_normal.pdf(X_comb_test_m4, mean=mean_same, cov=cov_same)
        scores_diff_test = scipy.stats.multivariate_normal.pdf(X_comb_test_m4, mean=mean_diff, cov=cov_diff)
        uncal_lr_test = np.divide(scores_same_test, scores_diff_test)

        biva_cal.fit(X=np.log10(uncal_lr_train), y=y_train)
        lrs_biva.append(biva_cal.transform(np.log10(uncal_lr_test)))

        results_to_save[count]['lrs_mfw'] = lrs_mfw[count].tolist()
        results_to_save[count]['lrs_voc'] = lrs_voc[count].tolist()
        results_to_save[count]['lrs_comb_a'] = lrs_comb_a[count].tolist()
        results_to_save[count]['lrs_comb_b'] = lrs_comb_b[count].tolist()
        results_to_save[count]['lrs_feat'] = lrs_features[count].tolist()
        results_to_save[count]['lrs_biva'] = lrs_biva[count].tolist()

        with open('frida/predictions/predictions_per_repeat.json', 'w') as fp:
            json.dump(results_to_save, fp)


    # calculate metrics for each method and log them
    mfw_res = calculate_metrics(lrs_mfw, y_all, full_list=all_metrics)
    voc_res = calculate_metrics(lrs_voc, y_all, full_list=all_metrics)
    lrs_multi = [np.multiply(lrs_mfw[i], lrs_voc[i]) for i in range(len(lrs_mfw))]
    lrs_multi_res = calculate_metrics(lrs_multi, y_all, full_list=all_metrics)
    comb_res_a = calculate_metrics(lrs_comb_a, y_all, full_list=all_metrics)
    comb_res_b = calculate_metrics(lrs_comb_b, y_all, full_list=all_metrics)
    feat_res = calculate_metrics(lrs_features, y_all, full_list=all_metrics)
    biva_res = calculate_metrics(lrs_biva, y_all, full_list=all_metrics)


    LOG.info(f'  mfw only: {mfw_res._fields} = {list(np.round(mfw_res, 3))}')
    LOG.info(f'  voc only: {voc_res._fields} = {list(np.round(voc_res, 3))}')
    LOG.info(f'  lrs comb by multi: {lrs_multi_res._fields} = {list(np.round(lrs_multi_res, 3))}')
    LOG.info(f'  lrs comb by logiA: {comb_res_a._fields} = {list(np.round(comb_res_a, 3))}')
    LOG.info(f'  lrs comb by logiB: {comb_res_b._fields} = {list(np.round(comb_res_b, 3))}')
    LOG.info(f'  lrs comb by featu: {feat_res._fields} = {list(np.round(feat_res, 3))}')
    LOG.info(f'  lrs comb by bivar: {biva_res._fields} = {list(np.round(biva_res, 3))}')

    return mfw_res, voc_res, lrs_multi_res, comb_res_a, comb_res_b, feat_res, biva_res


def calculate_eer(lrs, y):
    fpr, tpr, threshold = roc_curve(list(y), list(lrs), pos_label=1)
    fnr = 1 - tpr
    return fpr[np.nanargmin(np.absolute((fnr - fpr)))]


def calculate_metrics(lrs, y, full_list=False):
    """
    calculate several metrics

    :param lrs: lrs as derived by the evaluate_samesource
    :param y: true labels
    :param full_list: if True more metrics are calculated

    :return: [namedtuple] cllr, cllr_std, cllrmin, eer, eer_std, recall, precision (if full_list=True then
                          cllr, cllr_std, cllr_min, cllr_cal, eer, eer_std, accuracy, recall, true negative rate,
                          precision, mean_logLR_diff, mean_logLR_same)
    """

    cllrs = np.array([lir.metrics.cllr(lrs[i], y[i]) for i in range(len(lrs))])
    cllr = np.mean(cllrs)
    cllr_std = np.std(cllrs)

    cllrmin = np.mean(np.array([lir.metrics.cllr_min(lrs[i], y[i]) for i in range(len(lrs))]))

    eers = np.array([calculate_eer(lrs[i], y[i]) for i in range(len(lrs))])
    eer = np.mean(eers)
    eer_std = np.std(eers)

    recall = np.mean(np.array([np.mean(lrs[i][y[i] == 1] > 1) for i in range(len(lrs))]))  # true positive rate
    precision = np.mean(np.array([np.mean(y[i][lrs[i] > 1] == 1) for i in range(len(lrs))]))

    if full_list:
        cllrcal = cllr - cllrmin

        acc = np.mean(np.array([np.mean((lrs[i] > 1) == y[i]) for i in range(len(lrs))]))  # accuracy

        auc = np.mean(np.array([roc_auc_score(list(y[i]), list(lrs[i])) for i in range(len(lrs))]))

        tnr = np.mean(np.array([np.mean(lrs[i][y[i] == 0] < 1) for i in range(len(lrs))]))  # true negative rate
        mean_logLR_diff = np.mean(np.array([np.mean(np.log(lrs[i][y[i] == 0])) for i in range(len(lrs))]))
        mean_logLR_same = np.mean(np.array([np.mean(np.log(lrs[i][y[i] == 1])) for i in range(len(lrs))]))

        Metrics = collections.namedtuple('Metrics', ['cllr', 'cllr_std', 'cllrmin', 'cllrcal', 'eer', 'eer_std',
                                                     'accuracy', 'auc', 'recall', 'tnr', 'precision',
                                                     'mean_logLR_diff', 'mean_logLR_same'])
        res = Metrics(cllr, cllr_std, cllrmin, cllrcal, eer, eer_std, acc, auc, recall, tnr, precision,
                      mean_logLR_diff, mean_logLR_same)
    else:
        Metrics = collections.namedtuple('Metrics', ['cllr', 'cllr_std', 'cllrmin', 'eer', 'eer_std', 'recall',
                                                     'precision'])
        res = Metrics(cllr, cllr_std, cllrmin, eer, eer_std, recall, precision)

    return res


def aggregate_results(out_dir, results):
    '''
    prints resutls

    :param out_dir: [outdated at the moment]
    :param results: namedtuple (as return by the evaluate_samesource)

    '''
    for params, result in results:
        desc = ', '.join(f'{name}={value}' for name, value in params)
        print(f'mfw only: {result[0]._fields} = {list(np.round(result[0], 3))}--{desc}')
        print(f'voc only: {result[1]._fields} = {list(np.round(result[1], 3))}--{desc}')
        print(f'lrs comb by multi: {result[2]._fields} = {list(np.round(result[2], 3))}--{desc}')
        print(f'lrs comb by logiA: {result[3]._fields} = {list(np.round(result[3], 3))}--{desc}')
        print(f'lrs comb by logiB: {result[4]._fields} = {list(np.round(result[4], 3))}--{desc}')
        print(f'lrs comb by featu: {result[5]._fields} = {list(np.round(result[5], 3))}--{desc}')
        print(f'lrs comb by bivar: {result[6]._fields} = {list(np.round(result[6], 3))}--{desc}')


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
        # ('scale:standard', sklearn.preprocessing.StandardScaler()),
        ('pop:gauss', GaussianCdfTransformer()),  # cumulative density function for each feature
        # ('pop:gauss', sklearn.preprocessing.QuantileTransformer()),  # cumulative density function for each feature
    ])

    prep_kde = sklearn.pipeline.Pipeline([
        ('scale:standard', sklearn.preprocessing.StandardScaler()),
        ('pop:kde', KdeCdfTransformer()),  # cumulative density function for each feature
    ])

    ### CLASSIFIERS for authorship verification (a element-wise distance and a binaly ML alg is expected)

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
        ('clf:mlp', MLPClassifier(solver='adam', max_iter=800, alpha=0.001, hidden_layer_sizes=(5,), random_state=1)),
    ])

    exp = experiments.Evaluation(evaluate_samesource, partial(aggregate_results, resultdir))

    exp.parameter('dataset', dataset)
    exp.parameter('voc_data', voc_data)
    exp.parameter('device', 'telephone')  # options: telephone, headset, SM58close, AKGC400BL, SM58far

    exp.parameter('min_num_of_words', 0)

    exp.parameter('n_frequent_words', 200)
    exp.addSearch('n_frequent_words', [100, 200, 300], include_default=False)

    exp.parameter('max_n_of_pairs_per_class', 6000)
    exp.addSearch('max_n_of_pairs_per_class', [2500, 3000, 4000], include_default=False)

    exp.parameter('preprocessor', prep_gauss)

    exp.parameter('classifier', ('bray_logit', logit_br))
    exp.addSearch('classifier', [('br_mlp', br_mlp), ('bray_logit', logit_br)],
                  include_default=False)

    exp.parameter('calibrator', lir.LogitCalibrator())
    exp.parameter('all_metrics', False)
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
