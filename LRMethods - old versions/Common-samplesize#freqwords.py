#!/usr/bin/env python3


from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from itertools import combinations
from xgboost import XGBClassifier
import math as m
import pickle
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from Function_file import *
import lir as liar


#set parameters

#graphs
plt.rcParams.update({'figure.autolayout': True})
sns.set_style('whitegrid', {'axes.linewidth': 1, 'axes.edgecolor':'black'}) #0.15 is the grey variant
sns.set_palette('muted')
sns.set_context("paper", rc={"lines.linewidth": 1.5, "font.size": 18, "axes.labelsize":16, "axes.titlesize":18, "xtick.labelsize":14,
                             "ytick.labelsize":14, "legend.fontsize":16,}) #fontscale=1,5

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

hist_title = 'Histogram common LR scores'
ECE_title = 'ECE plot common LR'
PAV_title = 'PAV plot common LR'
Tippet_title = 'Tippett plot common LR'

#algorithm
test_authors = 10
repeat = 100
train_authors_vec = [170]
sample_size_total = [500]
n_freq_total = [50]
plotfigure = True
perauthor = True
train_samples = 3000
test_samples = 500
picklefile ='sLRcommon' + 'At' + str(test_authors) + 'Atr' + str(train_authors_vec) + 'rep' + str(repeat) + 'ss' + str(sample_size_total) + 'F' + str(n_freq_total)

LR_sLR_acc_overall = []
cllr_sLR_overall = []
LR_ACC_mean_sLR = []
cllr_mean_sLR = []
cllr_stat_sLR = []

blocks = []
LR_sLR_overall = []
labels_sLR_overall = []
labels_boxplot = []

for k_ss in train_authors_vec:
    for i_ss in sample_size_total:
        for j_ss in n_freq_total:

            cllr_sLR_tot = []
            LR_sLR_acc_tot = []

            LR_sLR_tot = []
            labels_sLR_tot = []
            timesLR = []

            sample_size = i_ss
            n_freq = j_ss
            train_authors = k_ss
            labels_boxplot.append(('F=' + str(n_freq) + ', N=' + str(sample_size)))

            if perauthor:
                speakers_path = 'JSON/speakers_author_specific_test.json'
                if os.path.exists(speakers_path):
                    print('loading', speakers_path)
                    speakers_wordlist = load_data(speakers_path)
                else:
                    speakers_wordlist = compile_data_author('SHA256_textfiles/sha256.filesnew_specific_test.txt')
                    store_data(speakers_path, speakers_wordlist)

            else:
                speakers_path = 'JSON/speakers_specific.json'
                if os.path.exists(speakers_path):
                    print('loading', speakers_path)
                    speakers_wordlist = load_data(speakers_path)
                else:
                    speakers_wordlist = compile_data('SHA256_textfiles/sha256.filesnew_specific.txt')
                    store_data(speakers_path, speakers_wordlist)

            wordlist = list(zip(*get_frequent_words(speakers_wordlist, n_freq)))[0]
            speakers = filter_texts_size_new(speakers_wordlist, wordlist, sample_size)
            speakers = dict(list(speakers.items()))
            X_temp, y_temp = to_vector_size(speakers, '0')
            author_uni = np.unique(y_temp)


            hist_fig = 'Hist_sLR_ss-F' + str(n_freq) + 'ss' + str(sample_size) + 'numbsamp' + str(k_ss)
            ECE_fig = 'ECE_sLR_ss-F' + str(n_freq) + 'ss' + str(sample_size) + 'numbsamp' + str(k_ss)
            PAV_fig = 'PAV_sLR_ss-F' + str(n_freq) + 'ss' + str(sample_size) + 'numbsamp' + str(k_ss)
            Tippet_fig = 'Tippett_sLR_ss-F' + str(n_freq) + 'ss' + str(sample_size) + 'numbsamp' + str(k_ss)

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

                scaler_pop = StandardScaler()
                norm_pop = Normalizer()
                X = norm_pop.fit_transform(X)
                scaler_pop.fit(X)

                norm_test = Normalizer()
                X_t = norm_test.fit_transform(X_t)

                # defining variance and mean arrays
                mean_pop = scaler_pop.mean_
                var_pop = scaler_pop.var_

                mean_source = collections.defaultdict()
                var_within = []
                for yvalue in np.unique(y):
                    scaler_source = StandardScaler()
                    Z = X[(y == yvalue)]
                    scaler_source.fit(Z)
                    mean_source[yvalue] = scaler_source.mean_
                    var_within.append([scaler_source.var_])

                var_within = np.reshape(np.asarray(var_within), (len(var_within), -1))
                var_source = np.mean(var_within, 0)

                features_ss, labels_ss = calc_LRfeat_ss(X, y, train_samples)
                features_ds, labels_ds = calc_LRfeat_ds(X, y, train_samples)
                features_ss_t, labels_ss_t = calc_LRfeat_ss(X_t, y_t, test_samples)
                features_ds_t, labels_ds_t = calc_LRfeat_ds(X_t, y_t, len(labels_ss_t))

                covar_pop = np.diag(var_pop)
                covar_source = np.diag(var_source)
                covar_pop_1 = np.diag(1 / var_pop)
                covar_source_1 = np.diag(1 / var_source)
                mean_col = np.reshape(np.transpose(mean_pop), (len(mean_pop), 1))

                scores_ss = []
                scores_ds = []
                scores_ss_t = []
                scores_ds_t = []

                # Same source scores
                likeli_denum = []
                likeli_num = []
                for i in features_ss:
                    Xcol = np.reshape(i, ((2 * n_freq), 1))
                    temp = calc_like(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq,
                                      Xcol[0:n_freq]) * calc_like(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq,
                                      Xcol[n_freq:(2 * n_freq)])
                    likeli_denum.append(temp)
                    likeli_num.append(calc_like_common(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq,
                                             Xcol))
                likeli_denum = np.asarray(likeli_denum)
                likeli_num = np.asarray(likeli_num)
                check = likeli_num / likeli_denum
                scores_ss.extend(check)

                # Different source scores
                likeli_denum = []
                likeli_num = []
                for i in features_ds:
                    Xcol = np.reshape(i, ((2 * n_freq), 1))
                    temp = calc_like(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq,
                                      Xcol[0:n_freq]) * calc_like(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq,
                                      Xcol[n_freq:(2 * n_freq)])
                    likeli_denum.append(temp)
                    likeli_num.append(calc_like_common(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq,
                                             Xcol))
                likeli_denum = np.asarray(likeli_denum)
                likeli_num = np.asarray(likeli_num)
                check = likeli_num / likeli_denum
                scores_ds.extend(check)

                # Test scores
                likeli_denum = []
                likeli_num = []
                for i in features_ss_t:
                    Xcol = np.reshape(i, ((2 * n_freq), 1))
                    temp = calc_like(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq,
                                      Xcol[0:n_freq]) * calc_like(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq,
                                      Xcol[n_freq:(2 * n_freq)])
                    likeli_denum.append(temp)
                    likeli_num.append(calc_like_common(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq,
                                             Xcol))
                likeli_denum = np.asarray(likeli_denum)
                likeli_num = np.asarray(likeli_num)
                check = likeli_num / likeli_denum
                scores_ss_t.extend(check)

                likeli_denum = []
                likeli_num = []
                for i in features_ds_t:
                    Xcol = np.reshape(i, ((2 * n_freq), 1))
                    temp = calc_like(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq,
                                      Xcol[0:n_freq]) * calc_like(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq,
                                      Xcol[n_freq:(2 * n_freq)])
                    likeli_denum.append(temp)
                    likeli_num.append(calc_like_common(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq,
                                             Xcol))
                likeli_denum = np.asarray(likeli_denum)
                likeli_num = np.asarray(likeli_num)
                check = likeli_num / likeli_denum
                scores_ds_t.extend(check)

                scores_test = np.log10(np.concatenate((scores_ss_t, scores_ds_t)))
                labels_test = list(map(int, (np.append(labels_ss_t, labels_ds_t, axis=0))))

                scores = np.log10(np.concatenate((scores_ss, scores_ds)))
                labels = list(map(int, (np.append(labels_ss, labels_ds, axis=0))))

                calibrator = liar.KDECalibrator()
                calibrator.fit(np.reshape(scores, (len(scores), 1)), labels)
                LR_common = calibrator.transform(np.asarray(scores_test))

                if plotfigure and step == 0:
                    liar.plotting.plot_score_distribution_and_calibrator_fit(calibrator, scores, labels,
                                                                             kw_figure={}, colorset=colors,
                                                                             titleplot=hist_title, savefig=hist_fig)

                LR_test = LR_acc(LR_common)  # naar 0 of 1
                accur_sLR = LR_acc_calc(LR_test, labels_test)
                print('accuracy LR common = ', accur_sLR)

                LR_common1, LR_common2 = liar.util.Xy_to_Xn(LR_common, labels_test)

                # CLLR
                cllr_sLR = liar.calculate_lr_statistics(LR_common1, LR_common2)

                print('Cllr common', cllr_sLR.cllr, '\nCllr_min common', cllr_sLR.cllr_min, '\nCllr_cal common',
                      cllr_sLR.cllr_cal)
                print('Cllr class0', cllr_sLR.cllr_class0, '\nCllr class1', cllr_sLR.cllr_class1)

                cllr_sLR_tot.append(cllr_sLR.cllr)
                LR_sLR_acc_tot.append(accur_sLR)
                LR_sLR_tot.append(LR_common)
                labels_sLR_tot.append(labels_test)
                cllr_stat_sLR.append(cllr_sLR)

            cllr_sLR_overall.append(cllr_sLR_tot)
            LR_sLR_acc_overall.append(LR_sLR_acc_tot)

            LR_ACC_mean_sLR.append(np.mean(LR_sLR_acc_tot))
            cllr_mean_sLR.append(np.mean(cllr_sLR_tot))

            LR_sLR_tot = np.concatenate(LR_sLR_tot)
            labels_sLR_tot = np.concatenate(labels_sLR_tot)

            LR_sLR_overall.append(LR_sLR_tot)
            labels_sLR_overall.append(labels_sLR_tot)

            # Tippett plot
            liar.plotting.plot_tippet(LR_sLR_tot, np.asarray(labels_sLR_tot), savefig=Tippet_fig, titleplot=Tippet_title)
            # PAV plot
            liar.plotting.plot_pav(LR_sLR_tot, np.asarray(labels_sLR_tot), savefig=PAV_fig, titleplot=PAV_title)
            # ECEplot
            liar.ece.plot(LR_sLR_tot, np.asarray(labels_sLR_tot), savefig=ECE_fig, titleplot=ECE_title)

    with open(picklefile, 'wb') as f:
        pickle.dump(
            [cllr_stat_sLR, cllr_mean_sLR, LR_sLR_acc_overall, LR_ACC_mean_sLR, labels_boxplot, LR_sLR_overall,
             labels_sLR_overall], f)

    print(labels_boxplot)
    print('sLR LR acc', LR_ACC_mean_sLR)
    print('sLR cllr', cllr_mean_sLR)

    fig = plt.figure(**{})
    ax = fig.add_subplot(121)
    plt.plot(sample_size_total, cllr_mean_sLR, label='Cllr')
    plt.xlabel('Samplesize (N)')
    plt.ylabel('Accuracy [%]')
    plt.legend()
    plt.title('sLR accuracy')
    ax = fig.add_subplot(122)
    plt.plot(sample_size_total, LR_ACC_mean_sLR, label='Accuracy')
    plt.xlabel('Samplesize (N)')
    plt.ylabel('Cllr')
    plt.title('sLR Cllr')
    plt.legend()
    plt.savefig('plot_Acccllr_sLR')
    plt.show()

    plot_boxplot(LR_sLR_acc_overall, labels_boxplot, xaxis='Samplesize (N)', yaxis='Accuracy',
                 boxtitle='Boxplot accuracy sLR', savfig='boxplot_acc_sLR', rotation=45)
    plot_boxplot(cllr_sLR_overall, labels_boxplot, xaxis='Samplesize (N)', yaxis='Cllr', boxtitle='Boxplot Cllr sLR',
                 savfig='boxplot_cllr_sLR', rotation=45)

