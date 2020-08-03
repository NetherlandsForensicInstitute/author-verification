#!/usr/bin/env python3


import lir as liar

from sklearn.preprocessing import StandardScaler, Normalizer

from Function_file import *

# set parameters and text files

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
ECE_title = 'ECE plot distance method'
PAV_title = 'PAV plot distance method'
Tippet_title = 'Tippett plot distance method'

# algorithms
repeat = 100
test_authors = 10
train_authors = 180
sample_size_total = [1400]
n_freq_total = [50, 150, 250]
plotfigure = True
CGN = False
train_samples = 5000
test_samples = 1000
picklefile = 'DistshanOWN' + 'At' + str(test_authors) + 'Atr' + str(train_authors) + 'rep' + str(
    repeat) + 'ss' + str(sample_size_total) + 'F' + str(n_freq_total) + 'S' + str(train_samples)

LR_shan_acc_overall = []
cllr_shan_overall = []
LR_ACC_mean_shan = []
cllr_mean_shan = []
cllr_stat_shan = []
LR_shan_overall = []
labels_shan_overall = []
labels_boxplot = []

speakers_path = 'JSON/speakers_FINAL.json'
if os.path.exists(speakers_path):
    print('loading', speakers_path)
    speakers_wordlist = load_data(speakers_path)
else:
    speakers_wordlist = compile_data('SHA256_textfiles/FINALdata.txt')
    store_data(speakers_path, speakers_wordlist)

for i_ss in sample_size_total:
    for j_ss in n_freq_total:

        cllr_shan_tot = []
        LR_shan_acc_tot = []

        LR_shan_tot = []
        labels_shan_tot = []

        sample_size = i_ss
        n_freq = j_ss
        labels_boxplot.append(('F=' + str(n_freq) + ', N=' + str(sample_size)))


        wordlist = list(zip(*get_frequent_words(speakers_wordlist, n_freq)))[0]
        speakers = filter_texts_size_new(speakers_wordlist, wordlist, sample_size)
        speakers = dict(list(speakers.items()))
        X_temp, y_temp = to_vector_size(speakers, '0')
        author_uni = np.unique(y_temp)
        for i in author_uni:
            if len(X_temp[y_temp == i]) < 2:
                print(str(i) + ':    ' + str(len(X_temp[y_temp == i])))
        hist_fig = 'Hist_dist_ss-F' + str(n_freq) + 'ss' + str(sample_size) + 'numbsamp'
        ECE_fig = 'ECE_dist_ss-F' + str(n_freq) + 'ss' + str(sample_size) + 'numbsamp'
        PAV_fig = 'PAV_dist_ss-F' + str(n_freq) + 'ss' + str(sample_size) + 'numbsamp'
        Tippet_fig = 'Tippett_dist_ss-F' + str(n_freq) + 'ss' + str(sample_size) + 'numbsamp'

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

            labels_ss, scores_ss = ss_score(X, y, 'shan', n_freq, train_samples)

            labels_ds, scores_ds = ds_score(X, y, 'shan', n_freq, train_samples)

            labels_ss_t, scores_ss_t = ss_score(X_t, y_t, 'shan', n_freq, test_samples)

            labels_ds_t, scores_ds_t = ds_score(X_t, y_t, 'shan', n_freq, min(len(labels_ss_t), test_samples))

            X_shan = np.concatenate((scores_ss, scores_ds))
            y_shan = list(map(int, (np.append(labels_ss, labels_ds, axis=0))))

            X_t_shan = np.concatenate((scores_ss_t, scores_ds_t))
            y_t_shan = list(map(int, (np.append(labels_ss_t, labels_ds_t, axis=0))))

            # LR calculation
            calibrator = liar.KDECalibrator()
            calibrator.fit(X_shan, y_shan)
            LR_shan = calibrator.transform(X_t_shan)

            LR_test, accur = LR_acc_calc(LR_shan, y_t_shan)  # naar 0 of 1

            # LR berekenen per class
            LR_sh1, LR_sh2 = liar.util.Xy_to_Xn(LR_shan, y_t_shan)

            # CLLR
            cllr_shan = liar.calculate_lr_statistics(LR_sh1, LR_sh2)

            #print('Cllr shan', cllr_shan.cllr, '\nCllr_min shan', cllr_shan.cllr_min, '\nCllr_cal shan', cllr_shan.cllr_cal)

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
             boxtitle='Boxplot accuracy distance shan own', savfig='boxplot_LRACC_distance_shan', rotation=45)
plot_boxplot(cllr_shan_overall, labels_boxplot, xaxis='Samplesize (N)', yaxis='Cllr',
             boxtitle='Boxplot Cllr distance shan own',
             savfig='boxplot_cllr_distance_shan', rotation=45)

fig = plt.figure(**{})
ax = fig.add_subplot(111)
plt.plot(n_freq_total, cllr_mean_shan, label='Cllr')
plt.plot(n_freq_total, LR_ACC_mean_shan, label='Accuracy')
plt.xlabel('Samplesize (N)')
plt.ylabel('Accuracy [%] / Cllr')
plt.legend()
plt.title('Distance method accuracy and Cllr')
plt.savefig('plot_Acccllr_distance')
plt.show()

fig = plt.figure(**{})
plt.plot(n_freq_total, cllr_mean_shan, label='Cllr', color=colors[0])
plt.xlabel('Samplesize (N)')
plt.ylabel('Accuracy [%]')
plt.legend()
plt.title('Distance accuracy')
plt.savefig('plot_Acccllr_distance')
plt.show()

fig = plt.figure(**{})
plt.plot(n_freq_total, LR_ACC_mean_shan, label='Accuracy', color=colors[1])
plt.xlabel('Samplesize (N)')
plt.ylabel('Cllr')
plt.title('Distance Cllr')
plt.legend()
plt.savefig('plot_Acc_distance')
plt.show()'''
