#!/usr/bin/env python3

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import lir as liar
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, Normalizer
from tqdm import tqdm

from authorship import get_data_orig as get_data
#from authorship import get_data_replacement as get_data

import Function_file as data

# set parameters

# graphs
plt.rcParams.update({'figure.autolayout': True})
sns.set_style('whitegrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})  # 0.15 is the grey variant
sns.set_palette('muted')
sns.set_context("paper", rc={"lines.linewidth": 1.5, "font.size": 18, "axes.labelsize": 16, "axes.titlesize": 18,
                             "xtick.labelsize": 14,
                             "ytick.labelsize": 14, "legend.fontsize": 16, })  # fontscale=1,5
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

hist_title = 'Histogram SVM scores'
ECE_title = 'ECE plot SVM'
PAV_title = 'PAV plot SVM'
Tippet_title = 'Tippett plot SVM'

# algorithm
repeat = 10
test_authors = 10
train_authors = 190
sample_size_total = [250, 750, 1500]
n_freq_total = [200]
plotfigure = True
CGN = False
train_samples = 5000
test_samples = 1000
picklefile = 'SVM' + 'At' + str(test_authors) + 'Atr' + str(train_authors) + 'rep' + str(repeat) + 'ss' + str(
    sample_size_total) + 'F' + str(n_freq_total)

LR_clf_acc_overall = []
cllr_clf_overall = []
LR_ACC_mean_clf = []
cllr_mean_clf = []
cllr_stat_clf = []

LR_clf_overall = []
labels_clf_overall = []
labels_boxplot = []


for i_ss in sample_size_total:
    for j_ss in n_freq_total:

        cllr_clf_tot = []
        LR_clf_acc_tot = []

        LR_clf_tot = []
        labels_clf_tot = []

        sample_size = i_ss
        n_freq = j_ss
        labels_boxplot.append(('F=' + str(n_freq) + ', N=' + str(sample_size)))

        X_temp, y_temp = get_data('data/ALLdata.txt', n_freq, sample_size)
        author_uni = np.unique(y_temp)

        hist_fig = 'Hist_SVM_ss-F' + str(n_freq) + 'ss' + str(sample_size)
        ECE_fig = 'ECE_SVM_ss-F' + str(n_freq) + 'ss' + str(sample_size)
        PAV_fig = 'PAV_SVM_ss-F' + str(n_freq) + 'ss' + str(sample_size)
        Tippet_fig = 'Tippett_SVM_ss-F' + str(n_freq) + 'ss' + str(sample_size)

        for step in tqdm(range(repeat)):
            randsample = random.sample(list(author_uni), (test_authors + train_authors))
            authors_t = np.asarray(randsample[0:test_authors])
            authors_tr = np.asarray(randsample[test_authors:(test_authors + train_authors)])

            X_t = []
            y_t = []
            X = []
            y = []

            for i in authors_t:
                X_t.append(np.array(X_temp[y_temp == i]))
                y_t.extend(y_temp[y_temp == i])
            X_t = np.concatenate(X_t)
            y_t = np.ravel(np.array(y_t))

            for i in authors_tr:
                X.append(np.array(X_temp[y_temp == i]))
                y.extend(y_temp[y_temp == i])
            X = np.concatenate(X)
            y = np.ravel(np.array(y))

            labels_ss, features_ss = data.ss_feature(X, y, 'shan', train_samples)
            labels_ds, features_ds = data.ds_feature(X, y, 'shan', train_samples)
            labels_ss_t, features_ss_t = data.ss_feature(X_t, y_t, 'shan', test_samples)
            labels_ds_t, features_ds_t = data.ds_feature(X_t, y_t, 'shan', min(len(labels_ss_t), test_samples))

            X = np.concatenate((features_ss, features_ds))
            y = list(map(int, (np.append(labels_ss, labels_ds, axis=0))))
            X = X.reshape(len(X), -1)

            X_t = np.concatenate((features_ss_t, features_ds_t))
            y_t = list(map(int, (np.append(labels_ss_t, labels_ds_t, axis=0))))
            if len(X_t.shape) == 3:
                X_t = X_t.reshape(len(X_t), -1)

            clf = SVC(gamma='scale', kernel='linear', probability=True, class_weight='balanced')
            clf.fit(X, y)

            calibrator1 = liar.KDECalibrator()

            cal_clf = clf.predict_proba(X)

            calibrator1.fit(cal_clf[:, 0], np.asarray(y))

            y_proba_clf = clf.predict_proba(X_t)

            LRtest_clf = calibrator1.transform(y_proba_clf[:, 0])

            y_LR_clf, accur_clf = data.LR_acc_calc(LRtest_clf, np.asarray(y_t))

            # LR berekenen
            LR_clf1, LR_clf2 = liar.util.Xy_to_Xn(LRtest_clf, np.asarray(y_t))

            # CLLR
            cllr_clf = liar.calculate_lr_statistics(LR_clf1, LR_clf2)

            cllr_clf_tot.append(cllr_clf.cllr)
            LR_clf_acc_tot.append(accur_clf)
            LR_clf_tot.append(LRtest_clf)
            labels_clf_tot.append(y_t)
            cllr_stat_clf.append(cllr_clf)

            if plotfigure and step == 0:
                liar.plotting.plot_score_distribution_and_calibrator_fit(calibrator1, cal_clf[:, 0], y,
                                                                         savefig=hist_fig)

        cllr_clf_overall.append(cllr_clf_tot)
        LR_clf_acc_overall.append(LR_clf_acc_tot)

        LR_ACC_mean_clf.append(np.mean(LR_clf_acc_tot))
        cllr_mean_clf.append(np.mean(cllr_clf_tot))

        LR_clf_tot = np.concatenate(LR_clf_tot)
        LR_clf_overall.append(LR_clf_tot)

        labels_clf_tot = np.concatenate(labels_clf_tot)
        labels_clf_overall.append(labels_clf_tot)

        # Tippett plot
        liar.plotting.plot_tippett(LR_clf_tot, labels_clf_tot, savefig=Tippet_fig)
        # PAV plot
        liar.plotting.plot_pav(LR_clf_tot, labels_clf_tot, savefig=PAV_fig)
        # ECEplot
        liar.ece.plot(LR_clf_tot, labels_clf_tot, path=ECE_fig)

with open(picklefile, 'wb') as f:
    pickle.dump([cllr_stat_clf, cllr_mean_clf, LR_clf_acc_overall, LR_ACC_mean_clf, labels_boxplot, LR_clf_overall,
                 labels_clf_overall], f)

print(labels_boxplot)
print('clf LR acc', LR_ACC_mean_clf)
print('clf cllr', cllr_mean_clf)

'''fig = plt.figure(**{})
ax = fig.add_subplot(111)
plt.plot(n_freq_total, cllr_mean_clf, label='Cllr')
plt.plot(n_freq_total, LR_ACC_mean_clf, label='Accuracy')
plt.xlabel('Samplesize (N)')
plt.ylabel('Accuracy [%] / Cllr')
plt.legend()
plt.title('SVM accuracy and Cllr')
plt.savefig('plot_Acccllr_SVM')
plt.show()

plot_boxplot(LR_clf_acc_overall, labels_boxplot, xaxis='Samplesize (N)', yaxis='Accuracy',
             boxtitle='Boxplot accuracy SVM', savfig='boxplot_LRACC_SVM')

plot_boxplot(cllr_clf_overall, labels_boxplot, xaxis='Samplesize (N)', yaxis='Cllr', boxtitle='Boxplot Cllr SVM',
             savfig='boxplot_cllr_SVM')'''
