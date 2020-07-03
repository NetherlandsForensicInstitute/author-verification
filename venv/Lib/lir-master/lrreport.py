#!/usr/bin/env python3

import math

import numpy as np
from sklearn.linear_model import LogisticRegression

import lir
from lir.plotting import NormalCllrEvaluator, ScoreBasedCllrEvaluator, makeplot_cllr


class generate_data:
    def __init__(self, loc, datasize):
        self.loc = loc
        self.datasize = datasize

    def __call__(self, x):
        return np.random.normal(loc=self.loc, size=(self.datasize, 1))


def plot_scheidbaarheid(repeat):
    xvalues = np.arange(0, 6, 1)
    generator_args = [ {
        'class0_train': lambda x: np.random.normal(loc=0, size=(100, 1)),
        'class1_train': lambda x: np.random.normal(loc=x, size=(100, 1)),
        'class0_calibrate': lambda x: np.random.normal(loc=0, size=(100, 1)),
        'class1_calibrate': lambda x: np.random.normal(loc=x, size=(100, 1)),
        'class0_test': lambda x: np.random.normal(loc=0, size=(100, 1)),
        'class1_test': lambda x: np.random.normal(loc=x, size=(100, 1)),
        'distribution_mean_delta': d,
        'repeat': repeat,
        } for d in xvalues ]

    generators = [
        NormalCllrEvaluator('baseline', 0, 1, 0, 1),
        ScoreBasedCllrEvaluator('logit/fraction', LogisticRegression(solver='lbfgs'), lir.ScalingCalibrator(lir.FractionCalibrator()), []),
        ScoreBasedCllrEvaluator('logit/kde', LogisticRegression(solver='lbfgs'), lir.KDECalibrator(), []),
        ScoreBasedCllrEvaluator('logit/gauss', LogisticRegression(solver='lbfgs'), lir.GaussianCalibrator(), []),
        ScoreBasedCllrEvaluator('logit/copy', LogisticRegression(solver='lbfgs'), lir.DummyCalibrator(), []),
    ]

    makeplot_cllr('dx', generators, list(zip(xvalues, generator_args)), savefig='plot_scheidbaarheid.png', show=True)


def plot_datasize(repeat):
    xvalues = range(0, 7)
    dx = 1
    generator_args = []
    for x in xvalues:
        datasize = int(math.pow(2, x))
        generator_args.append({
            'class0_train': generate_data(0, datasize),
            'class1_train': generate_data(dx, datasize),
            'class0_calibrate': generate_data(0, 100),
            'class1_calibrate': generate_data(dx, 100),
            'class0_test': generate_data(0, 100),
            'class1_test': generate_data(dx, 100),
            'repeat': repeat,
        })

    generators = [
        NormalCllrEvaluator('baseline', 0, 1, dx, 1),
        ScoreBasedCllrEvaluator('logit/fraction', LogisticRegression(solver='lbfgs'), lir.ScalingCalibrator(lir.FractionCalibrator()), []),
        ScoreBasedCllrEvaluator('logit/kde', LogisticRegression(solver='lbfgs'), lir.KDECalibrator(), []),
        ScoreBasedCllrEvaluator('logit/gauss', LogisticRegression(solver='lbfgs'), lir.GaussianCalibrator(), []),
        ScoreBasedCllrEvaluator('logit/copy', LogisticRegression(solver='lbfgs'), lir.DummyCalibrator(), []),
    ]

    makeplot_cllr('data size 2^x; {repeat}x'.format(repeat=repeat), generators, list(zip(xvalues, generator_args)), savefig='plot_datasize.png', show=True)


def plot_split(repeat):
    datasize = 10
    testsize = 100
    dx = 1
    experiments = [
        ('split50', {
            'class0_train': lambda x: np.random.normal(loc=0, size=(int(datasize/2), 1)),
            'class1_train': lambda x: np.random.normal(loc=dx, size=(int(datasize/2), 1)),
            'class0_calibrate': lambda x: np.random.normal(loc=0, size=(int(datasize/2), 1)),
            'class1_calibrate': lambda x: np.random.normal(loc=dx, size=(int(datasize/2), 1)),
            'class0_test': lambda x: np.random.normal(loc=0, size=(testsize, 1)),
            'class1_test': lambda x: np.random.normal(loc=dx, size=(testsize, 1)),
            'repeat': repeat,
        }),
        ('2fold', {
            'class0_train': lambda x: np.random.normal(loc=0, size=(datasize, 1)),
            'class1_train': lambda x: np.random.normal(loc=dx, size=(datasize, 1)),
            'class0_test': lambda x: np.random.normal(loc=0, size=(testsize, 1)),
            'class1_test': lambda x: np.random.normal(loc=dx, size=(testsize, 1)),
            'train_folds': 2,
            'repeat': repeat,
        }),
        ('4fold', {
            'class0_train': lambda x: np.random.normal(loc=0, size=(datasize, 1)),
            'class1_train': lambda x: np.random.normal(loc=dx, size=(datasize, 1)),
            'class0_test': lambda x: np.random.normal(loc=0, size=(testsize, 1)),
            'class1_test': lambda x: np.random.normal(loc=dx, size=(testsize, 1)),
            'train_folds': 4,
            'repeat': repeat,
        }),
        ('reuse', {
            'class0_train': lambda x: np.random.normal(loc=0, size=(datasize, 1)),
            'class1_train': lambda x: np.random.normal(loc=dx, size=(datasize, 1)),
            'class0_test': lambda x: np.random.normal(loc=0, size=(testsize, 1)),
            'class1_test': lambda x: np.random.normal(loc=dx, size=(testsize, 1)),
            'train_reuse': True,
            'repeat': repeat,
        }),
    ]

    generators = [
        NormalCllrEvaluator('baseline', 0, 1, dx, 1),
        ScoreBasedCllrEvaluator('logit/fraction', LogisticRegression(solver='lbfgs'), lir.ScalingCalibrator(lir.FractionCalibrator()), []),
        ScoreBasedCllrEvaluator('logit/kde', LogisticRegression(solver='lbfgs'), lir.KDECalibrator(), []),
        ScoreBasedCllrEvaluator('logit/gauss', LogisticRegression(solver='lbfgs'), lir.GaussianCalibrator(), []),
        ScoreBasedCllrEvaluator('logit/copy', LogisticRegression(solver='lbfgs'), lir.DummyCalibrator(), []),
    ]

    makeplot_cllr('data splits of {} samples for each class'.format(datasize), generators, experiments, savefig='plot_split.png', show=True)


if __name__ == '__main__':
    plot_scheidbaarheid(20)
    plot_datasize(20)
    plot_split(20)
