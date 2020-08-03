"""
Dataset generators.
"""
import numpy as np


class NormalGenerator:
    """
    Samples scores from two independent normal distributions: one score
    distribution for H1 and one for H2. From each score an LR is calculated.
    """

    def __init__(self, mu0, sigma0, mu1, sigma1):
        """
        Initializes the generator with two normal distributions.

        :param mu0: mean of class 0 scores (H2)
        :param sigma0: standard deviation of class 0 scores (H2)
        :param mu1: mean of class 1 scores (H1)
        :param sigma1: standard deviation of class 1 scores (H1)
        """
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.mu1 = mu1
        self.sigma1 = sigma1

    def _get_probability(X, mu, sigma):
        return np.exp(-np.power(X - mu, 2) / (2*sigma*sigma)) / np.sqrt(2*np.pi*sigma*sigma)

    def sample_lrs(self, n0, n1):
        """
        Samples LRs from both distribtions.

        :param n0: number of LRs from class 0 (H2)
        :param n1: number of LRs from class 1 (H1)
        :returns: an array of LRs and an array of labels (value 0 or 1)
        """
        y = np.concatenate([np.zeros(n0), np.ones(n1)])
        X = np.concatenate([np.random.normal(loc=self.mu0, scale=self.sigma0, size=n0),
                            np.random.normal(loc=self.mu1, scale=self.sigma1, size=n1)])
        p0 = NormalGenerator._get_probability(X, self.mu0, self.sigma0)
        p1 = NormalGenerator._get_probability(X, self.mu1, self.sigma1)
        lr = p1 / p0

        return lr, y


class RandomFlipper:
    """
    Random mutilation of a dataset.

    TODO: this class is broken
    """
    def __init__(self, base_generator, p):
        self.gen = base_generator
        self.p = p

    def sample_lrs(self, n0, n1):
        lr, y = self.gen.sample_lrs(n0, n1)
        y[np.random.randint(0, len(y), int(self.p*n0))] = 0
        y[np.random.randint(0, len(y), int(self.p*n1))] = 1

        return lr, y
