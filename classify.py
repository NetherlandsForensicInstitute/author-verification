#!/usr/bin/env python3

import collections
import logging
import warnings

import confidence
import lir.multiclass as lir
import numpy as np
import scipy.stats
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection

from schotresten.database.db_connection import DBConnection


LOG = logging.getLogger(__name__)


class DataSource:
    def __init__(self, con, relevance_classes, ammo_types=None, invert_ammo_types=False):
        self._con = con
        self.relevance_classes = relevance_classes
        self._ammo_types = ammo_types
        self._invert_ammo_types = invert_ammo_types

    def ammo_types(self, ammo_types, invert_ammo_types=False):
        return DataSource(self._con, self.relevance_classes, ammo_types, invert_ammo_types)

    def get(self):
        with self._con.cursor() as cur:
            cur.execute('SET seed TO 0')

            qclasses_sum = [ f'SUM(CASE WHEN relevance_class = %s THEN 1 ELSE 0 END) class_{c}' for c in self.relevance_classes ]
            qclasses_select = [ f'class_{c}' for c in self.relevance_classes ]
            qwhere = [
                "stub.project = 'HULZEN'",
                "stub.sample_type = 'cartridge'",
                "NOT ammo_type = 'HULZEN_unknown'",
            ]

            qargs = []
            qargs.extend(self.relevance_classes)

            if self._ammo_types is not None:
                args_ammo_types = ','.join(['%s' for c in self._ammo_types])
                qwhere.append(f'ammo_type {"NOT" if self._invert_ammo_types else ""} IN ({args_ammo_types})')
                qargs.extend(self._ammo_types)

            q = f'''
                WITH hulzen_stub AS (
                    SELECT stub.ammo_type, RANDOM() rnd,
                        count(*) aantal_deeltjes,
                        {','.join(qclasses_sum)}
                    FROM stub JOIN particle ON particle.stub_id = stub.id
                    WHERE {' AND '.join(qwhere)}
                    GROUP BY stub.id, stub.ammo_type ORDER BY RANDOM())
                SELECT ammo_type, {','.join(qclasses_select)}
                FROM hulzen_stub
                '''
            cur.execute(q, qargs)
            LOG.debug(cur.query.decode('utf8'))

            rows = cur.fetchall()
            ammo_types = sorted(set([row[0] for row in rows]))

            X = np.array([row[1:] for row in rows])
            y = np.array([ammo_types.index(row[0]) for row in rows])

            return X, y


class ParticleCountToFraction(sklearn.base.TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        return (X.T / np.sum(X, axis=1)).T


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
    def fit(self, X):
        return self

    def transform(self, X, y):
        pairing = np.array(np.meshgrid(np.arange(X.shape[0]), np.arange(X.shape[0]))).T.reshape(-1, 2)  # generate all possible pairs
        pairing[pairing[:, 0] != pairing[:, 1], :]  # remove pairs with same id
        X = np.stack([X[pairing[:, 0]], X[pairing[:, 1]]], axis=2)  # pair instances by adding another dimension
        y = (y[pairing[:, 0]] == y[pairing[:, 1]]).astype(int)  # apply the new labels: 1=same_source versus 0=different_source

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
    def fit(self, X):
        return self

    def transform(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == 2

        return np.abs(X[:,:,0] - X[:,:,1])


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


def evaluate_samesource(clf, ds):
    X, y = ds.get()
    assert X.shape[0] > 0

    X = ParticleCountToFraction().fit_transform(X)  # replace particle counts by fractions

    # fit and apply a cumulative density function for each feature on a background dataset
    cdf = GaussianCdfTransformer()
    X = cdf.fit_transform(X)

    # pair instances: same source and different source
    X_pairs, y_pairs = InstancePairing().transform(X, y)

    # calculate the differences of the cumulative density values between instances of the same pairs
    X_pairs = AbsDiffTransformer().transform(X_pairs)

    # fit a classifier on the cumulative density differences of all features within pairs
    clf.fit(X_pairs, y_pairs)

    # calculate LRs
    lrs = clf.predict_lr(X_pairs)

    print('  average LR by class:', lir.metrics.by_class(lir.metrics.geometric_mean, lrs, y_pairs))
    print(f'  cllr: {lir.metrics.macro(lir.metrics.cllr, lrs, y_pairs)}')
    print()


def run(conn: DBConnection):
    ds = DataSource(conn, relevance_classes=['PbSbBa', 'PbSbBaSn', 'ZnTiGd', 'PbSb', 'PbSbSn', 'BaSb', 'BaSbSn', 'PbBa', 'Pb', 'Ba', 'Sb', 'ZnTi'])

    print(f'number of classes: {np.unique(ds.get()[1]).size}')
    print(f'number of instances: {ds.get()[1].size}')
    print()

    print('specific source feature based gaussian')
    evaluate_specificsource(lir.CalibratedScorer(GaussianScorer(), lir.DummyCalibrator()), ds)

    print('specific source feature based gaussian; calibrated')
    evaluate_specificsource(lir.CalibratedScorer(GaussianScorer(), lir.BalancedPriorCalibrator(lir.LogitCalibrator())), ds)

    print('same source score based gaussian')
    evaluate_samesource(lir.CalibratedScorer(LogisticRegression(class_weight='balanced'), lir.DummyCalibrator()), ds)

    print('same source score based gaussian; calibrated')
    evaluate_samesource(lir.CalibratedScorer(LogisticRegression(class_weight='balanced'), lir.LogitCalibrator()), ds)


if __name__ == '__main__':
    config = confidence.load_name('schotresten', 'local')
    warnings.filterwarnings("error")
    run(DBConnection(**config.database.schotresten))
