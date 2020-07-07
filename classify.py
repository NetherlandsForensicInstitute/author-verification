#!/usr/bin/env python3

import collections
import logging
import os
import warnings

import confidence
import lir
from matplotlib import pyplot as plt
import numpy as np
import scipy.spatial
import scipy.stats
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection
import sklearn.neighbors
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm

import Function_file as data


LOG = logging.getLogger(__name__)


class DataSource:
    def __init__(self, n_frequent_words, tokens_per_sample):
        self._n_freqwords = n_frequent_words
        self._tokens_per_sample = tokens_per_sample

    def get(self):
        os.makedirs('cache', exist_ok=True)
        speakers_path = 'cache/speakers_author.json'
        if os.path.exists(speakers_path):
            speakers_wordlist = data.load_data(speakers_path)
        else:
            speakers_wordlist = data.compile_data('SHA256_textfiles/sha256.filesnew.txt')
            data.store_data(speakers_path, speakers_wordlist)

        wordlist = [ word for word, freq in data.get_frequent_words(speakers_wordlist, self._n_freqwords) ]
        speakers = data.filter_texts_size_new(speakers_wordlist, wordlist, self._tokens_per_sample)
        X, y = data.to_vector_size(speakers)

        return X, y


class ParticleCountToFraction(sklearn.base.TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        return (X.T / np.sum(X, axis=1)).T


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
            feature_values = X[:,i]
            lower, upper = self.get_range(feature_values)

            kernel = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=.1).fit(feature_values.reshape(-1, 1))
            precomputed_values = np.arange(self._resolution+1).reshape(-1, 1) / self._resolution * (upper-lower) + lower
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
            feature_values = X[:,i]
            lower, upper = self.get_range(feature_values)

            percentiles = self._kernels[i][((feature_values - lower) / (upper-lower) * self._resolution).astype(int)]
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
        X = X[:,self._valid_features]
        return scipy.stats.norm.cdf(X, self._mean, self._std)


class InstancePairing(sklearn.base.TransformerMixin):
    def __init__(self, same_source_limit=None, different_source_limit=None):
        self._ss_limit = same_source_limit
        self._ds_limit = different_source_limit

    def fit(self, X):
        return self

    def transform(self, X, y):
        pairing = np.array(np.meshgrid(np.arange(X.shape[0]), np.arange(X.shape[0]))).T.reshape(-1, 2)  # generate all possible pairs
        same_source = y[pairing[:, 0]] == y[pairing[:, 1]]

        rows_same = np.where((pairing[:, 0] != pairing[:, 1]) & same_source)[0]  # pairs with different id and same source
        if self._ss_limit is not None and rows_same.size > self._ss_limit:
            rows_same = np.random.choice(rows_same, self._ss_limit, replace=False)

        rows_diff = np.where((pairing[:, 0] != pairing[:, 1]) & ~same_source)[0]  # pairs with different id and different source
        ds_limit = rows_diff.size if self._ds_limit is None else rows_same.size if self._ds_limit == 'balance' else self._ds_limit
        if rows_diff.size > ds_limit:
            rows_diff = np.random.choice(rows_diff, ds_limit, replace=False)

        pairing = np.concatenate([pairing[rows_same,:], pairing[rows_diff,:]])
        X = np.stack([X[pairing[:, 0]], X[pairing[:, 1]]], axis=2)  # pair instances by adding another dimension
        y = np.concatenate([np.ones(rows_same.size), np.zeros(rows_diff.size)])  # apply the new labels: 1=same_source versus 0=different_source

        return X, y


GaussParams = collections.namedtuple('StandardParams', ['mean0', 'std0', 'mean1', 'std1'])
class GaussianScorer(sklearn.base.BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        assert np.all(np.arange(np.max(y)+1) == np.unique(y)), 'classes must be numbered 0..n and all classes must occur at least once'

        self._classes = np.unique(y)
        self._models = {}
        for cls in self._classes:
            X0 = X[y!=cls]
            X1 = X[y==cls]

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
                p.append(p1/(p0+p1))

        return np.prod(lir.to_odds(np.array(p)), axis=2).T  # multiply the odds over categories (assume conditional independence)


class AbsDiffTransformer(sklearn.base.TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == 2

        return np.abs(X[:,:,0] - X[:,:,1])


class BrayDistance(sklearn.base.TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == 2

        left = X[:,:,0]
        right = X[:,:,1]

        return np.abs(right - left) / (np.abs(right + left) + 1)


class ShanDistance(sklearn.base.TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == 2

        p = X[:,:,0]
        q = X[:,:,1]
        p = p / np.sum(p, axis=0)
        q = q / np.sum(q, axis=0)
        m = (p + q) / 2.0
        left = scipy.spatial.distance.rel_entr(p, m)
        right = scipy.spatial.distance.rel_entr(q, m)
        result = np.sqrt((left + right) / 2.0)
        assert X.shape[0:2] == result.shape
        return result


def evaluate_specificsource(clf, ds):
    X, y = ds.get()
    assert X.shape[0] > 0

    X = ParticleCountToFraction().fit_transform(X)  # replace particle counts by fractions

    clf.fit(X, y)
    y_predicted = clf.predict(X)
    print(f'  best class score: {np.sum(y_predicted == y)} of {y.size}')

    lrs = clf.predict_lr(X)
    print('  macro average LR:', lir.metrics.macro(lir.metrics.geometric_mean, lrs, y))
    print(f'  cllr: {lir.metrics.micro(lir.metrics.cllr, lrs, y)}')
    print()


def get_pairs(X, y, authors_subset, ds_limit):
    X_subset = X[np.isin(y, authors_subset), :]
    y_subset = y[np.isin(y, authors_subset)]

    # pair instances: same source and different source
    return InstancePairing(different_source_limit=ds_limit).transform(X_subset, y_subset)


def evaluate_samesource(clf, ds, preprocessor):
    desc_pre = '; '.join(name for name, tr in preprocessor.steps)
    desc_clf = '; '.join(name for name, tr in clf.scorer.steps)
    print(f'same source: {desc_pre}; {desc_clf}')

    X, y = ds.get()
    assert X.shape[0] > 0

    # apply a preprocessor
    X = preprocessor.fit_transform(X)

    authors = np.unique(y)
    authors_train, authors_test = sklearn.model_selection.train_test_split(authors, test_size=.2, random_state=1)

    X_train, y_train = get_pairs(X, y, authors_train, 'balance')
    X_test, y_test = get_pairs(X, y, authors_test, 'balance')

    # fit a classifier on the cumulative density differences of all features within pairs
    clf.fit(X_train, y_train)

    # calculate LRs
    lrs = clf.predict_lr(X_test)

    print(f'  counts by class: diff={y_test.size-np.sum(y_test):.0f}; same={np.sum(y_test):.0f}')
    print(f'  average LR by class: 1/{np.exp(np.mean(-np.log(lrs[y_test==0])))}; {np.exp(np.mean(np.log(lrs[y_test==1])))}')
    print(f'  cllr: {lir.metrics.cllr(lrs, y_test)}')
    print()


def run():
    ds = DataSource(n_frequent_words=200, tokens_per_sample=750)

    print(f'number of classes: {np.unique(ds.get()[1]).size}')
    print(f'number of instances: {ds.get()[1].size}')
    print()

    prep_simple = sklearn.pipeline.Pipeline([
            ('scale:fraction', ParticleCountToFraction()),
        ])

    prep_gauss = sklearn.pipeline.Pipeline([
            ('scaler', sklearn.preprocessing.StandardScaler()),
            ('pop:gauss', GaussianCdfTransformer()),  # cumulative density function for each feature
        ])

    prep_kde = sklearn.pipeline.Pipeline([
            ('scaler', sklearn.preprocessing.StandardScaler()),
            ('pop:kde', KdeCdfTransformer()),  # cumulative density function for each feature
        ])

    clf = sklearn.pipeline.Pipeline([
            ('diff:abs', AbsDiffTransformer()),
            #('shan', ShanDistance()),
            #('bray', BrayDistance()),
            ('clf:logit', LogisticRegression(class_weight='balanced')),
            #('svc', sklearn.svm.SVC(probability=True)),
        ])

    svc = sklearn.pipeline.Pipeline([
            ('diff:abs', AbsDiffTransformer()),
            ('svc', sklearn.svm.SVC(probability=True)),
        ])

    evaluate_samesource(lir.CalibratedScorer(clf, lir.LogitCalibrator()), ds, prep_simple)
    evaluate_samesource(lir.CalibratedScorer(clf, lir.KDECalibrator(bandwidth=.1)), ds, prep_gauss)
    evaluate_samesource(lir.CalibratedScorer(clf, lir.KDECalibrator(bandwidth=.1)), ds, prep_kde)
    evaluate_samesource(lir.CalibratedScorer(svc, lir.KDECalibrator(bandwidth=.1)), ds, prep_kde)


if __name__ == '__main__':
    config = confidence.load_name('authorship', 'local')
    warnings.filterwarnings("error")
    np.random.seed(0)
    run()
