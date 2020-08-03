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

hist_title = 'Histogram specific LR scores'
ECE_title = 'ECE plot specific LR'
PAV_title = 'PAV plot specific LR'
Tippet_title = 'Tippett plot specific LR'

#algorithm
spec_samples = 5
test_authors = 10
repeat = 100
train_authors_vec = [170]
sample_size_total = [500]
n_freq_total = [20, 30, 40, 50]
plotfigure = True
perauthor = True
train_samples = 3000
test_samples = 500
picklefile ='sLRspec' + 'At' + str(test_authors) + 'Atr' + str(train_authors_vec) + 'rep' + str(repeat) + 'ss' + str(sample_size_total) + 'F' + str(n_freq_total)

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


            for i in author_uni:
                if len(X_temp[y_temp == i]) <7:
                    print(str(i) + ':    ' + str(len(X_temp[y_temp == i])))


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
                X_spec_temp = []
                X_spec_t = []
                y_spec_t = []
                X_trainss = []
                y_trainss = []

                for i in authors_t:
                    X_spec_temp = np.array(X_temp[y_temp == i])
                    X_spec_t.append(np.asarray(X_spec_temp[0:spec_samples]).reshape(-1, n_freq))
                    y_spec_t.extend(spec_samples * [i])
                    X_t.append(np.asarray(X_spec_temp[spec_samples:len(X_spec_temp)]).reshape(-1, n_freq))
                    y_t.extend((len(X_spec_temp) - spec_samples) * [i])

                X_t = np.concatenate(X_t)
                y_t = np.ravel(np.array(y_t))
                X_spec_t = np.concatenate(X_spec_t)
                y_spec_t = np.ravel(np.array(y_spec_t))

                for i in authors_tr:
                    X_train_temp = np.array(X_temp[y_temp == i])
                    X_trainss.append(np.asarray(X_train_temp[0:spec_samples]).reshape(-1, n_freq))
                    y_trainss.extend(spec_samples * [i])
                    X.append(np.asarray(X_train_temp[spec_samples:len(X_train_temp)]).reshape(-1, n_freq))
                    y.extend((len(X_train_temp) - spec_samples) * [i])

                X_trainss = np.concatenate(X_trainss)
                y_trainss = np.ravel(np.array(y_trainss))
                X = np.concatenate(X)
                y = np.ravel(np.array(y))

                scaler_pop = StandardScaler()
                norm_pop = Normalizer()
                X = norm_pop.fit_transform(X)
                scaler_pop.fit(X)

                norm_train = Normalizer()
                X_trainss = norm_train.fit_transform(X_trainss)

                norm_spec = Normalizer()
                X_spec_t = norm_spec.fit_transform(X_spec_t)

                norm_test = Normalizer()
                X_t = norm_test.fit_transform(X_t)

                # defining variance and mean arrays
                mean_pop = scaler_pop.mean_
                var_pop = scaler_pop.var_

                mean_source = collections.defaultdict()
                var_within = []
                for yvalue in np.unique(y_trainss):
                    scaler_source = StandardScaler()
                    Z = X_trainss[(y_trainss == yvalue)]
                    scaler_source.fit(Z)
                    mean_source[yvalue] = scaler_source.mean_
                    var_within.append([scaler_source.var_])

                mean_source_t = collections.defaultdict()
                for yvalue in np.unique(y_spec_t):
                    scaler_source = StandardScaler()
                    Z = X_spec_t[(y_spec_t == yvalue)]
                    scaler_source.fit(Z)
                    mean_source_t[yvalue] = scaler_source.mean_


                var_within = np.reshape(np.asarray(var_within), (len(var_within), -1))
                var_source = np.mean(var_within, 0)

                covar_pop = np.diag(var_pop)
                covar_source = np.diag(var_source)
                covar_pop_1 = np.diag(1 / var_pop)
                covar_source_1 = np.diag(1 / var_source)
                mean_col = np.reshape(np.transpose(mean_pop), (len(mean_pop), 1))

                labels_ss = []
                scores_ss = []
                labels_ds = []
                scores_ds = []


                # Same source scores
                for yvalue in np.unique(y):
                    rv = stats.multivariate_normal(mean=mean_source[yvalue], cov=var_source)
                    likeli_num = rv.pdf(X[y == yvalue])

                    likeli_denum = []
                    Xrow = X[y == yvalue]
                    for i in Xrow:
                        Xcol = np.reshape(i, (len(i), 1))
                        temp = calc_like(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq, Xcol)
                        likeli_denum.append(temp)
                    likeli_denum = np.asarray(likeli_denum)
                    check = likeli_num / likeli_denum
                    scores_ss.extend(check)
                    if isinstance(likeli_num, float):
                        labels_ss.extend('1')
                    else:
                        labels_ss.extend('1' * len(likeli_num))


                # Different source scores
                for yvalue in np.unique(y):
                    Xset = X[y != yvalue]
                    set = random.sample(range(len(Xset)), 30)
                    Xset = X[set]
                    rv = stats.multivariate_normal(mean=mean_source[yvalue], cov=var_source)
                    likeli_num = rv.pdf(Xset)
                    likeli_denum = []
                    for i in Xset:
                        Xcol = np.reshape(i, (len(i), 1))
                        temp = calc_like(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq, Xcol)
                        likeli_denum.append(temp)
                    likeli_denum = np.asarray(likeli_denum)
                    check = likeli_num / likeli_denum
                    scores_ds.extend(check)
                    labels_ds.extend('0' * len(likeli_num))

                trainingsample_ss = random.sample(range(len(labels_ss)), min(len(labels_ss), train_samples))
                trainingsample = random.sample(range(len(labels_ds)), train_samples)

                labels_ss = [labels_ss[i] for i in trainingsample_ss]
                scores_ss = [scores_ss[i] for i in trainingsample_ss]

                labels_ds = [labels_ds[i] for i in trainingsample]
                scores_ds = [scores_ds[i] for i in trainingsample]

                scores = np.log10(np.concatenate((scores_ss, scores_ds)))
                labels = list(map(int, (np.append(labels_ss, labels_ds, axis=0))))

                print(len(labels))


                labels_ss_t = []
                scores_ss_t = []
                labels_ds_t = []
                scores_ds_t = []

                for yuni in np.unique(y_spec_t):

                    # TEST Same source scores
                    rv = stats.multivariate_normal(mean=mean_source_t[yuni], cov=var_source)
                    likeli_num = rv.pdf(X_t[y_t == yuni])
                    Xrow = X_t[y_t == yuni]
                    likeli_denum = []
                    for i in Xrow:
                        Xcol = np.reshape(i, (len(i), 1))
                        temp = calc_like(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq, Xcol)
                        likeli_denum.append(temp)
                    likeli_denum = np.asarray(likeli_denum)
                    LR_app_ss = likeli_num / likeli_denum
                    scores_ss_t.extend(LR_app_ss)
                    if isinstance(likeli_num, float):
                        labels_ss_t.extend('1')
                    else:
                        labels_ss_t.extend('1' * len(likeli_num))


                    # TEST Different source scores
                    likeli_num = rv.pdf(X_t[y_t != yuni])
                    Xrow = X_t[y_t != yuni]
                    likeli_denum = []
                    for i in Xrow:
                        Xcol = np.reshape(i, (len(i), 1))
                        temp = calc_like(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq, Xcol)
                        likeli_denum.append(temp)
                    likeli_denum = np.asarray(likeli_denum)
                    LR_app_ds = likeli_num / likeli_denum

                    testsample = random.sample(range(len(LR_app_ds)), min(len(LR_app_ss),len(LR_app_ds)))
                    LR_app_ds = [LR_app_ds[i] for i in testsample]
                    scores_ds_t.extend(LR_app_ds)
                    labels_ds_t.extend('0' * len(LR_app_ds))

                scores_test = np.log10(np.concatenate((scores_ss_t, scores_ds_t)))
                labels_test = list(map(int, (np.append(labels_ss_t, labels_ds_t, axis=0))))
                print(len(labels_test))
                calibrator = liar.KDECalibrator()
                calibrator.fit(np.reshape(scores, (len(scores), 1)), labels)
                LR_spec = calibrator.transform(np.asarray(scores_test))


                if plotfigure and step==0:
                    liar.plotting.plot_score_distribution_and_calibrator_fit(calibrator, scores, labels,
                                                                             kw_figure={}, colorset=colors,
                                                                             titleplot=hist_title, savefig=hist_fig)

                LR_test = LR_acc(LR_spec)  # naar 0 of 1
                accur_sLR = LR_acc_calc(LR_test, labels_test)
                print('accuracy LR specific = ', accur_sLR)
                print('LR values = ', LR_spec)


                LR_spec1, LR_spec2 = liar.util.Xy_to_Xn(LR_spec, labels_test)

                # CLLR
                cllr_sLR = liar.calculate_lr_statistics(LR_spec1, LR_spec2)

                print('Cllr spec', cllr_sLR.cllr, '\nCllr_min spec', cllr_sLR.cllr_min, '\nCllr_cal spec', cllr_sLR.cllr_cal)
                print('Cllr class0', cllr_sLR.cllr_class0, '\nCllr class1', cllr_sLR.cllr_class1)

                cllr_sLR_tot.append(cllr_sLR.cllr)
                LR_sLR_acc_tot.append(accur_sLR)
                LR_sLR_tot.append(LR_spec)
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
            [cllr_stat_sLR, cllr_mean_sLR, LR_sLR_acc_overall, LR_ACC_mean_sLR, labels_boxplot, LR_sLR_overall, labels_sLR_overall], f)

    print('sLR LR acc', LR_ACC_mean_sLR)
    print('sLR cllr', cllr_mean_sLR)

    fig = plt.figure(**{})
    ax = fig.add_subplot(111)
    plt.plot(sample_size_total, cllr_mean_sLR, label='Cllr')
    plt.plot(sample_size_total, LR_ACC_mean_sLR, label='Accuracy')
    plt.xlabel('Samplesize (N)')
    plt.ylabel('Accuracy [%] / Cllr')
    plt.legend()
    plt.title('sLRoost accuracy and Cllr')
    plt.savefig('plot_Acccllr_sLRoost')
    plt.show()

    plot_boxplot(LR_sLR_acc_overall, labels_boxplot, xaxis='Samplesize (N)', yaxis='Accuracy',
                 boxtitle='Boxplot accuracy sLR', savfig='boxplot_acc_sLR', rotation=45)
    plot_boxplot(cllr_sLR_overall, labels_boxplot, xaxis='Samplesize (N)', yaxis='Cllr', boxtitle='Boxplot Cllr sLR',
                 savfig='boxplot_cllr_sLR', rotation=45)

