#!/usr/bin/env python3

import lir as liar
from sklearn.svm import SVC

from utils import *

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

# Parameter definition
repeat = 100
test_authors = 10
train_authors = 190
sample_size_total = [250, 500, 750, 1000, 1250, 1500]
n_freq_total = [200]
plotfigure = True
CGN = False
train_samples = 5000
test_samples = 1000
picklefile = f'SVM_At_{test_authors}_Atr_{train_authors}_rep_{repeat}_ss_{sample_size_total}_F_{n_freq_total}'

# Results
LR_clf_acc_overall = []
cllr_clf_overall = []
LR_ACC_mean_clf = []
cllr_mean_clf = []
cllr_stat_clf = []

LR_clf_overall = []
labels_clf_overall = []
labels_boxplot = []

# data loading
config = load_data('config.yaml', 'yaml')
input_path = config['speakers_path_CGN'] if CGN else config['speakers_path']

if os.path.exists(input_path['json']):
    print('loading', input_path['json'])
    speakers_wordlist = load_data(input_path['json'])
else:
    speakers_wordlist = compile_data(input_path['txt'])
    store_data(input_path['json'], speakers_wordlist)

# Experiments
for i_ss in sample_size_total:
    for j_ss in n_freq_total:

        cllr_clf_tot = []
        LR_clf_acc_tot = []

        LR_clf_tot = []
        labels_clf_tot = []

        sample_size = i_ss
        n_freq = j_ss
        labels_boxplot.append(f'F={n_freq}, N={sample_size}')

        wordlist = list(zip(*get_n_most_frequent_words(speakers_wordlist.values(), n_freq)))[0]
        speakers = filter_texts_size_new(speakers_wordlist, wordlist, sample_size)
        X_temp, y_temp = to_vector_size(speakers)
        speakers_unique = np.unique(y_temp)

        hist_fig = 'Hist_SVM_ss-F' + str(n_freq) + 'ss' + str(sample_size)
        ECE_fig = 'ECE_SVM_ss-F' + str(n_freq) + 'ss' + str(sample_size)
        PAV_fig = 'PAV_SVM_ss-F' + str(n_freq) + 'ss' + str(sample_size)
        Tippet_fig = 'Tippett_SVM_ss-F' + str(n_freq) + 'ss' + str(sample_size)

        for step in tqdm(range(repeat)):
            np.random.shuffle(speakers_unique)
            speakers_train = speakers_unique[test_authors:]
            speakers_test = speakers_unique[0:test_authors]

            X_train = []
            y_train = []
            X_test = []
            y_test = []

            for i in speakers_train:
                X_train.append(np.array(X_temp[y_temp == i]))
                y_train.extend(y_temp[y_temp == i])
            X_train = np.concatenate(X_train)
            y_train = np.ravel(np.array(y_train))

            for i in speakers_test:
                X_test.append(np.array(X_temp[y_temp == i]))
                y_test.extend(y_temp[y_temp == i])
            X_test = np.concatenate(X_test)
            y_test = np.ravel(np.array(y_test))

            labels_ss, features_ss = ss_feature(X_train, y_train, 'shan', train_samples)
            labels_ds, features_ds = ds_feature(X_train, y_train, 'shan', train_samples)
            labels_ss_t, features_ss_t = ss_feature(X_test, y_test, 'shan', test_samples)
            labels_ds_t, features_ds_t = ds_feature(X_test, y_test, 'shan', min(len(labels_ss_t), test_samples))

            X_train = np.concatenate((features_ss, features_ds))
            y_train = list(map(int, (np.append(labels_ss, labels_ds, axis=0))))
            X_train = X_train.reshape(len(X_train), -1)

            X_test = np.concatenate((features_ss_t, features_ds_t))
            y_test = list(map(int, (np.append(labels_ss_t, labels_ds_t, axis=0))))
            if len(X_test.shape) == 3:
                X_test = X_test.reshape(len(X_test), -1)

            clf = SVC(gamma='scale', kernel='linear', probability=True, class_weight='balanced')
            clf.fit(X_train, y_train)

            calibrator = liar.KDECalibrator()

            cal_clf = clf.predict_proba(X_train)

            calibrator.fit(cal_clf[:, 0], y_train)

            y_proba_clf = clf.predict_proba(X_test)

            LRtest_clf = calibrator.transform(y_proba_clf[:, 0])

            y_LR_clf, accur_clf = LR_acc_calc(LRtest_clf, y_test)

            # LR berekenen
            LR_clf1, LR_clf2 = liar.util.Xy_to_Xn(LRtest_clf, y_test)

            # CLLR
            cllr_clf = liar.calculate_lr_statistics(LR_clf1, LR_clf2)

            cllr_clf_tot.append(cllr_clf.cllr)
            LR_clf_acc_tot.append(accur_clf)
            LR_clf_tot.append(LRtest_clf)
            labels_clf_tot.append(y_test)
            cllr_stat_clf.append(cllr_clf)

            if plotfigure and step == 0:
                liar.plotting.plot_score_distribution_and_calibrator_fit(calibrator, cal_clf[:, 0], y_train,
                                                                         kw_figure={}, colorset=colors,
                                                                         titleplot=hist_title, savefig=hist_fig)

        cllr_clf_overall.append(cllr_clf_tot)
        LR_clf_acc_overall.append(LR_clf_acc_tot)

        LR_ACC_mean_clf.append(np.mean(LR_clf_acc_tot))
        cllr_mean_clf.append(np.mean(cllr_clf_tot))

        LR_clf_tot = np.concatenate(LR_clf_tot)
        LR_clf_overall.append(LR_clf_tot)

        labels_clf_tot = np.concatenate(labels_clf_tot)
        labels_clf_overall.append(labels_clf_tot)

        # Tippett plot
        liar.plotting.plot_tippet(LR_clf_tot, labels_clf_tot, savefig=Tippet_fig, titleplot=Tippet_title)
        # PAV plot
        liar.plotting.plot_pav(LR_clf_tot, labels_clf_tot, savefig=PAV_fig, titleplot=PAV_title)
        # ECEplot
        liar.ece.plot(LR_clf_tot, labels_clf_tot, savefig=ECE_fig, titleplot=ECE_title)

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
