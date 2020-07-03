"""
Empirical Cross Entrpy (ECE)

The discrimination and calibration of the LRs reported by some systems can also
be measured separately. The empirical cross entropy (ECE) plot is a graphical
way of doing this.

The ECE is the average of -P(Hp) * log2(P(Hp|LRi)) for all LRi when Hp is true,
and -P(Hd) * log2(P(Hd|LRi)) for all LRi when Hd is true.

See:
[-] D. Ramos, Forensic evidence evaluation using automatic speaker recognition
    systems. Ph.D. Thesis. Universidad Autonoma de Madrid.
[-] Bernard Robertson, G.A. Vignaux and Charles Berger, Interpreting Evidence:
    Evaluating Forensic Science in the Courtroom, 2nd edition, 2016, pp. 96-97.
"""
import matplotlib.pyplot as plt
import numpy as np

from .calibration import IsotonicCalibrator
from . import util


def plot(lrs, y, log_prior_odds_range=None, on_screen=False, savefig=None, titleplot=None, kw_figure={}):
    """
    Generates an ECE plot for a set of LRs and corresponding ground-truth
    labels.

    The x-axis indicates the log prior odds of a sample being drawn from class
    1; the y-axis shows the entropy for (1) a non-informative system (dotted
    line), (2) the set of LR values (line), and (3) the set of LR values after
    PAV-transformation (Pool Adjacent Violators, dashed line).

    :param lrs: an array of LRs
    :param y: an array of ground-truth labels (values 0 for Hd or 1 for Hp);
        must be of the same length as `lrs`
    :param log_prior_odds_range: the range of prior odds (tuple of two values,
        indicating both ends of the range on the x-axis)
    :param on_screen: boolean, show plot on screen interactively
    :param path: path name or None, write plot to file as PNG image (default
        None)
    :param kw_figure: dict of parameters to pass to `plt.figure()`
    """
    if log_prior_odds_range is None:
        log_prior_odds_range = (-3, 3)

    log_prior_odds = np.arange(*log_prior_odds_range, .01)
    prior_odds = np.power(10, log_prior_odds)

    fig = plt.figure(**kw_figure)

    # plot reference
    plt.plot(log_prior_odds, calculate_ece(np.ones(len(lrs)), y, util.to_probability(prior_odds)), linestyle=':', label='reference')

    # plot LRs
    plt.plot(log_prior_odds, calculate_ece(lrs, y, util.to_probability(prior_odds)), linestyle='-', label='LRs')

    # plot PAV LRs
    pav_lrs = IsotonicCalibrator().fit_transform(util.to_probability(lrs), y)
    plt.plot(log_prior_odds, calculate_ece(pav_lrs, y, util.to_probability(prior_odds)), linestyle='--', label='PAV LRs')

    plt.xlabel("prior log10 odds")
    plt.ylabel("empirical cross-entropy")
    plt.ylim((0,None))
    plt.xlim(log_prior_odds_range)
    plt.legend()
    plt.grid(True)
    if titleplot is not None:
        plt.title(titleplot)
    if on_screen:
        plt.show()
    if savefig is not None:
        plt.savefig(savefig)

    plt.close(fig)


def calculate_ece(lrs, y, priors):
    """
    Calculates the empirical cross-entropy (ECE) of a set of LRs and
    corresponding ground-truth labels.

    An entropy is calculated for each element of `priors`.

    :param lrs: an array of LRs
    :param y: an array of ground-truth labels of the LRs (values 0 for Hd or 1
        for Hp); must be of the same length as `lrs`.
    :param priors: an array of prior probabilities of the samples being drawn
        from class 1 (values in range [0..1])
    :returns: an array of entropy values of the same length as `priors`
    """
    
    prior_odds = np.repeat(util.to_odds(priors), len(lrs)).reshape((len(priors), len(lrs)))
    posterior_odds = prior_odds * lrs
    posterior_p = util.to_probability(posterior_odds)

    with np.errstate(invalid='ignore'):
        ece0 = - (1 - priors.reshape((len(priors),1))) * np.log2(1 - posterior_p[:,y == 0])
        ece1 = -      priors.reshape((len(priors),1))  * np.log2(    posterior_p[:,y == 1])

    ece0[np.isnan(ece0)] = np.inf
    ece1[np.isnan(ece1)] = np.inf

    avg0 = np.average(ece0, axis=1)
    avg1 = np.average(ece1, axis=1)

    return avg0 + avg1
