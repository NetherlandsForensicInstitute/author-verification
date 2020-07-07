#!/usr/bin/env python3


import lir as liar
import random
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, Normalizer

from Function_file import *

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

hist_title = 'Histogram distance scores'
ECE_title = 'ECE plot distance'
PAV_title = 'PAV plot distance'
Tippet_title = 'Tippett plot distance'

# algorithms
repeat = 100
test_authors = 10
train_authors = 190
train_samples_vec = [10, 50, 75, 100, 200, 300, 500, 1000, 2000, 3000, 4000]
sample_size_total = [500]
n_freq_total = [200]
plotfigure = True
test_samples = 1000
picklefile = 'Distshan' + 'At' + str(test_authors) + 'Atr' + str(train_authors) + 'rep' + str(repeat) + 'ss' + str(
    sample_size_total) + 'F' + str(n_freq_total) + 'numb' + str(train_samples_vec)

LR_shan_acc_overall = []
cllr_shan_overall = []
LR_ACC_mean_shan = []
cllr_mean_shan = []
cllr_stat_shan = []
blocks = []
LR_shan_overall = []
labels_shan_overall = []
labels_boxplot = []

for k_ss in train_samples_vec:
    for i_ss in sample_size_total:
        for j_ss in n_freq_total:

            cllr_shan_tot = []
            LR_shan_acc_tot = []

            LR_shan_tot = []
            labels_shan_tot = []
            sample_size = i_ss
            n_freq = j_ss
            train_samples = k_ss

            labels_boxplot.append('S#=' + str(train_samples))

            speakers_path = 'JSON/speakers_author.json'
            if os.path.exists(speakers_path):
                print('loading', speakers_path)
                speakers_wordlist = load_data(speakers_path)
            else:
                speakers_wordlist = compile_data('SHA256_textfiles/sha256.filesnew.txt')
                store_data(speakers_path, speakers_wordlist)

            wordlist = list(zip(*get_frequent_words(speakers_wordlist, n_freq)))[0]
            speakers = filter_texts_size_new(speakers_wordlist, wordlist, sample_size)
            speakers = dict(list(speakers.items()))
            X_temp, y_temp = to_vector_size(speakers)
            author_uni = np.unique(y_temp)

            hist_fig = 'Hist_dist_ss-F' + str(n_freq) + 'ss' + str(sample_size) + 'numbsamp' + str(k_ss)
            ECE_fig = 'ECE_dist_ss-F' + str(n_freq) + 'ss' + str(sample_size) + 'numbsamp' + str(k_ss)
            PAV_fig = 'PAV_dist_ss-F' + str(n_freq) + 'ss' + str(sample_size) + 'numbsamp' + str(k_ss)
            Tippet_fig = 'Tippett_dist_ss-F' + str(n_freq) + 'ss' + str(sample_size) + 'numbsamp' + str(k_ss)

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

                scaler = Normalizer()
                X = scaler.fit_transform(X)
                X_t = scaler.fit_transform(X_t)

                scaler_std = StandardScaler()
                scaler_std.fit(X)

                labels_ds, scores_ds = ds_score(X, y, 'shan', n_freq, train_samples)

                labels_ss, scores_ss = ss_score(X, y, 'shan', n_freq, train_samples)

                labels_ss_t, scores_ss_t = ss_score(X_t, y_t, 'shan', n_freq, test_samples)

                labels_ds_t, scores_ds_t = ds_score(X_t, y_t, 'shan', n_freq, min(len(labels_ss_t), test_samples))


                X_shan = np.concatenate((scores_ss, scores_ds))
                y_shan = np.asarray(map(int, (np.append(labels_ss, labels_ds, axis=0))))

                X_t_shan = np.concatenate((scores_ss_t, scores_ds_t))
                y_t_shan = list(map(int, (np.append(labels_ss_t, labels_ds_t, axis=0))))

                calibrator = liar.KDECalibrator()
                calibrator.fit(X_shan, y_shan)
                LR_shan = calibrator.transform(X_t_shan)

                LR_test, accur = LR_acc_calc(LR_shan, y_t_shan)  # naar 0 of 1

                # LR berekenen
                LR_sh1, LR_sh2 = liar.util.Xy_to_Xn(LR_shan, y_t_shan)

                # CLLR
                cllr_shan = liar.calculate_lr_statistics(LR_sh1, LR_sh2)

                cllr_shan_tot.append(cllr_shan.cllr)
                LR_shan_acc_tot.append(accur)
                LR_shan_tot.append(LR_shan)
                labels_shan_tot.append(y_t_shan)
                cllr_stat_shan.append(cllr_shan)

                if plotfigure and step == 0:
                    liar.plotting.plot_score_distribution_and_calibrator_fit(calibrator, X_shan, y_shan,
                                                                             kw_figure={}, colorset=colors,
                                                                             titleplot=hist_title, savefig=hist_fig)

            cllr_shan_overall.append(cllr_shan_tot)
            LR_shan_acc_overall.append(LR_shan_acc_tot)
            LR_ACC_mean_shan.append(np.mean(LR_shan_acc_tot))
            cllr_mean_shan.append(np.mean(cllr_shan_tot))

            LR_shan_tot = np.concatenate(LR_shan_tot)
            LR_shan_overall.append(LR_shan_tot)
            labels_shan_tot = np.concatenate(labels_shan_tot)
            labels_shan_overall.append(labels_shan_tot)

# Tippett plot
liar.plotting.plot_tippet(LR_shan_tot, labels_shan_tot, savefig=Tippet_fig, titleplot=Tippet_title)
# PAV plot
liar.plotting.plot_pav(LR_shan_tot, labels_shan_tot, savefig=str(PAV_fig + 'shan'),
                       titleplot=str(PAV_title + 'shan'))
# ECEplot
liar.ece.plot(LR_shan_tot, labels_shan_tot, savefig=ECE_fig, titleplot=ECE_title)

print(labels_boxplot)
print('shan LR acc', LR_ACC_mean_shan)
print('shan cllr', cllr_mean_shan)

with open(picklefile, 'wb') as f:
    pickle.dump(
        [cllr_stat_shan, cllr_mean_shan, LR_shan_acc_overall, LR_ACC_mean_shan, labels_boxplot, LR_shan_overall,
         labels_shan_overall], f)

'''plot_boxplot(LR_shan_acc_overall, labels_boxplot, xaxis='Samplesize (N)', yaxis='Accuracy',
             boxtitle='Boxplot accuracy distance shan', savfig='boxplot_LRACC_distance_shan', rotation=45)
plot_boxplot(cllr_shan_overall, labels_boxplot, xaxis='Samplesize (N)', yaxis='Cllr',
             boxtitle='Boxplot Cllr distance shan',
             savfig='boxplot_cllr_distance_shan', rotation=45)
             
fig = plt.figure(**{})
plt.plot(train_samples_vec, LR_ACC_mean_shan, label='Cllr', color=colors[0])
plt.xlabel('Number of samples')
plt.ylabel('Accuracy [%]')
plt.legend()
plt.title('Distance accuracy')
plt.savefig('plot_Acc_distance')
plt.show()

fig = plt.figure(**{})
plt.plot(train_samples_vec, cllr_mean_shan, label='Accuracy', color=colors[1])
plt.xlabel('Number of samples')
plt.ylabel('Cllr')
plt.title('Distance Cllr')
plt.legend()
plt.savefig('plot_cllr_distance')
plt.show()'''
