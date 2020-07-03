#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, Normalizer
from xgboost import XGBClassifier
import pickle

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

from Function_file import *

import liar as liar

if __name__ == '__main__':
    speakers_path = 'speakers_1vs1.json'
    if os.path.exists(speakers_path):
        print('loading', speakers_path)
        speakers = load_data(speakers_path)
    else:
        speakers = compile_data('sha256_1vs1.txt')
        store_data(speakers_path, speakers)

    speakers_path = 'speakers.json'
    if os.path.exists(speakers_path):
        print('loading', speakers_path)
        speakers_wordlist = load_data(speakers_path)
    else:
        speakers_wordlist = compile_data('sha256.filesnew.txt')
        store_data(speakers_path, speakers_wordlist)

    sample_size = 50
    n_freq = 150

    wordlist = list(zip(*get_frequent_words(speakers_wordlist, n_freq)))[0]

    speakers = filter_texts_size(speakers, wordlist, sample_size)
    speakers = dict(list(speakers.items()))

    X, y = to_vector_size(speakers, '0')

    scaler = Normalizer()
    X = scaler.fit_transform(X)

    z = 0
    for i in range(len(y) - 1):
        if y[i] == y[i + 1]:
            y[i] = z
        else:
            y[i] = z
            z = z + 1
    y[len(y) - 1] = y[len(y) - 2]


y = list(map(int, y))
if len(X.shape) == 3:
    X_t_final = X.reshape(len(X), -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.9, test_size=.1)

'''X_train_nb = X_train
X_test_nb = X_test
# X_cal_nb = X_cal
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# X_cal = scaler.transform(X_cal)'''


clf = SVC(gamma='scale', kernel='linear', probability=True, class_weight='balanced')
scores_clf = cross_val_score(clf, X_train, y_train, cv=10)
clf.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
scores_rf = cross_val_score(rf, X_train, y_train, cv=10)
rf.fit(X_train, y_train)

mnb = MultinomialNB()
scores_nb = cross_val_score(mnb, X_train, y_train, cv=10)
mnb.fit(X_train, y_train)

XGB = XGBClassifier(learning_rate=0.2, max_depth = 3)
XGB.fit(X_train, y_train)
scores_xgb = cross_val_score(XGB, X_train, y_train, cv=10)


print('Support Vector score:', np.mean(scores_clf))
print('Naive Bayes (M) score:', np.mean(scores_nb))
print('Random forest score:', np.mean(scores_rf))
print('XGB score:', np.mean(scores_xgb))

calibrator = liar.KDECalibrator()
calibrator2 = liar.KDECalibrator()
calibrator3 = liar.KDECalibrator()
calibrator4 = liar.KDECalibrator()

X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, train_size=.5, test_size=.5)

clf = SVC(gamma='scale', kernel='linear', probability=True)
clf.fit(X_train, y_train)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

cal_clf = clf.predict_proba(X_cal)
cal_rf = rf.predict_proba(X_cal)
cal_mnb = mnb.predict_proba(X_cal)
cal_xgb = XGB.predict_proba(X_cal)

calibrator.fit(cal_clf[:, 0], y_cal)
calibrator2.fit(cal_mnb[:, 0], y_cal)
calibrator3.fit(cal_rf[:, 0], y_cal)
calibrator4.fit(cal_xgb[:, 0], y_cal)

y_proba_clf = clf.predict_proba(X_test)
y_proba_mnb = mnb.predict_proba(X_test)
y_proba_rf = rf.predict_proba(X_test)
y_proba_xgb = XGB.predict_proba(X_test)

y_pred_clf = clf.predict(X_test)
y_pred_mnb = mnb.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_xgb = XGB.predict(X_test)

LRtest_clf = calibrator.transform(y_proba_clf[:, 0])
LRtest_mnb = calibrator2.transform(y_proba_mnb[:, 0])
LRtest_rf = calibrator3.transform(y_proba_rf[:, 0])
LRtest_xgb = calibrator4.transform(y_proba_xgb[:, 0])

# plot hist
X1, X2 = liar.util.Xy_to_Xn(X_cal, y_cal)
X1, X2 = liar.util.Xy_to_Xn(X_cal, y_cal)
histcal_clf1 = clf.predict_proba(X1)
histcal_clf2 = clf.predict_proba(X2)
histcal_rf1 = rf.predict_proba(X1)
histcal_rf2 = rf.predict_proba(X2)
histcal_mnb1 = mnb.predict_proba(X1)
histcal_mnb2 = mnb.predict_proba(X2)
histcal_xgb1 = XGB.predict_proba(X1)
histcal_xgb2 = XGB.predict_proba(X2)
X_plot = np.linspace(0, 1, 100)[:, np.newaxis]

plt.figure(figsize=(10, 10))
plt.hist([histcal_clf1[:, 0], histcal_clf2[:, 0]], bins=15, density=True, color=['seagreen', 'tomato'])
dens = calibrator._kde0.score_samples(X_plot)
dens1 = calibrator._kde1.score_samples(X_plot)
plt.fill_between(X_plot[:, 0], np.exp(dens).transpose(), alpha=0.3, color='darkgreen')
plt.plot(X_plot[:, 0], np.exp(dens).transpose(), color='darkgreen')
plt.fill_between(X_plot[:, 0], np.exp(dens1).transpose(), alpha=0.3, color='darkred')
plt.plot(X_plot[:, 0], np.exp(dens1).transpose(), color='darkred')
plt.title('SVM')
plt.show()

plt.figure(figsize=(10, 10))
plt.hist([histcal_rf1[:, 0], histcal_rf2[:, 0]], bins=15, density=True, color=['seagreen', 'tomato'])
dens_rf = calibrator3._kde0.score_samples(X_plot)
dens1_rf = calibrator3._kde1.score_samples(X_plot)
plt.fill_between(X_plot[:, 0], np.exp(dens_rf).transpose(), alpha=0.3, color='darkgreen')
plt.plot(X_plot[:, 0], np.exp(dens_rf).transpose(), color='darkgreen')
plt.fill_between(X_plot[:, 0], np.exp(dens1_rf).transpose(), alpha=0.3, color='darkred')
plt.plot(X_plot[:, 0], np.exp(dens1_rf).transpose(), color='darkred')
plt.title('RF')
plt.show()

plt.figure(figsize=(10, 10))
plt.hist([histcal_mnb1[:, 0], histcal_mnb2[:, 0]], bins=15, density=True, color=['seagreen', 'tomato'])
dens_nb = calibrator2._kde0.score_samples(X_plot)
dens1_nb = calibrator2._kde1.score_samples(X_plot)
plt.fill_between(X_plot[:, 0], np.exp(dens_nb).transpose(), alpha=0.3, color='darkgreen')
plt.plot(X_plot[:, 0], np.exp(dens_nb).transpose(), color='darkgreen')
plt.fill_between(X_plot[:, 0], np.exp(dens1_nb).transpose(), alpha=0.3, color='darkred')
plt.plot(X_plot[:, 0], np.exp(dens1_nb).transpose(), color='darkred')
plt.title('MNB')
plt.show()

plt.figure(figsize=(10, 10))
plt.hist([histcal_xgb1[:, 0], histcal_xgb2[:, 0]], bins=15, density=True, color=['seagreen', 'tomato'])
dens = calibrator4._kde0.score_samples(X_plot)
dens1 = calibrator4._kde1.score_samples(X_plot)
plt.fill_between(X_plot[:, 0], np.exp(dens).transpose(), alpha=0.3, color='darkgreen')
plt.plot(X_plot[:, 0], np.exp(dens).transpose(), color='darkgreen')
plt.fill_between(X_plot[:, 0], np.exp(dens1).transpose(), alpha=0.3, color='darkred')
plt.plot(X_plot[:, 0], np.exp(dens1).transpose(), color='darkred')
plt.title('XGB')
plt.show()

'''# score LR density plot
plt.figure(figsize=(10, 10))
loglr = dens - dens1
plt.plot(X_plot[:, 0], loglr)
plt.show()

plt.figure(figsize=(10, 10))
loglr = dens_rf - dens1_rf
plt.plot(X_plot[:, 0], loglr)
plt.show()

plt.figure(figsize=(10, 10))
loglr = dens_nb - dens1_nb
plt.plot(X_plot[:, 0], loglr)
plt.show()'''

# LR berekenen
LR_rf1, LR_rf2 = liar.util.Xy_to_Xn(LRtest_rf, y_test)
LR_mnb1, LR_mnb2 = liar.util.Xy_to_Xn(LRtest_mnb, y_test)
LR_clf1, LR_clf2 = liar.util.Xy_to_Xn(LRtest_clf, y_test)
LR_xgb1, LR_xgb2 = liar.util.Xy_to_Xn(LRtest_xgb, y_test)


# CLLR
cllr_rf = liar.calculate_cllr(LR_rf1, LR_rf2)
cllr_mnb = liar.calculate_cllr(LR_mnb1, LR_mnb2)
cllr_clf = liar.calculate_cllr(LR_clf1, LR_clf2)
cllr_xgb = liar.calculate_cllr(LR_xgb1, LR_xgb2)


print('Cllr rf', cllr_rf.cllr)
print('Cllr clf', cllr_clf.cllr)
print('Cllr mnb', cllr_mnb.cllr)
print('Cllr xgb', cllr_xgb.cllr)

''''# Tippett plot
arr = np.log10(np.concatenate((LRtest_clf, LRtest_mnb, LRtest_rf)))
min = np.min(arr) - 1
max = np.max(arr) + 1

plt.figure(figsize=(10, 10))
xplot = np.linspace(min, max, 100)

perc1 = (np.sum(i > xplot for i in np.log10(LR_rf1)) / len(LR_rf1)) * 100
perc2 = (np.sum(i > xplot for i in np.log10(LR_rf2)) / len(LR_rf2)) * 100

plt.plot(xplot, perc1, color='g', label='RF')
plt.plot(xplot, perc2, color='g')

perc1 = (np.sum(i > xplot for i in np.log10(LR_clf1)) / len(LR_clf1)) * 100
perc2 = (np.sum(i > xplot for i in np.log10(LR_clf2)) / len(LR_clf2)) * 100

plt.plot(xplot, perc1, color='r', label='SVM')
plt.plot(xplot, perc2, color='r')

perc1 = (np.sum(i > xplot for i in np.log10(LR_mnb1)) / len(LR_mnb1)) * 100
perc2 = (np.sum(i > xplot for i in np.log10(LR_mnb2)) / len(LR_mnb2)) * 100

plt.plot(xplot, perc1, color='b', label='NB')
plt.plot(xplot, perc2, color='b')

plt.xlabel('Log likelihood ratio')
plt.ylabel('Cumulative proportion')

plt.legend()
plt.savefig('TestTippett.png')
plt.show()

# ECEplot
liar.ece.plot(LRtest_rf, np.asarray(y_test), path='testECE_rf')
liar.ece.plot(LRtest_clf, np.asarray(y_test), path='testECE_clf')
liar.ece.plot(LRtest_mnb, np.asarray(y_test), path='testECE_mnb')

# PAV plot
liar.pav.plot(LRtest_rf, np.asarray(y_test), path='testpav_rf')
liar.pav.plot(LRtest_clf, np.asarray(y_test), path='testpav_clf')
liar.pav.plot(LRtest_mnb, np.asarray(y_test), path='testpav_mnb')

scorer = liarm.CalibratedScorer(scorer=LG, calibrator=calibrator, fit_calibrator=True)
scorer_nb = liarm.CalibratedScorer(scorer=mnb, calibrator=calibrator, fit_calibrator=True)
scorer_clf = liarm.CalibratedScorer(scorer=clf, calibrator=calibrator, fit_calibrator=True)

cal = calibrator.fit(X_cal, y_cal)
cal_nb = calibrator.fit(X_cal_nb, y_cal)
cal_clf = calibrator.fit(X_cal, y_cal)


scorer.fit(X_cal, y_cal)
scorer_nb.fit(X_cal_nb, y_cal)
scorer_clf.fit(X_cal, y_cal)


p = scorer.predict_proba(X_test, y_test)
p_nb = scorer_nb.predict_proba(X_test
_nb, y_test)
p_clf = scorer_clf.predict_proba(X_test, y_test)


plt.figure(figsize=(15, 15))
liarm.plot_density(scorer.calibrator, show=True)
liarm.plot_lr_histogram(p, y_test, show=True)

plt.figure(figsize=(15, 15))
liarm.plot_density(scorer_nb.calibrator, show=True)
liarm.plot_lr_histogram(p_nb, y_test, show=True)


plt.figure(figsize=(15, 15))
liarm.plot_density(scorer_clf.calibrator, show=True)
liarm.plot_lr_histogram(p_clf, y_test, show=True)



score = liarm.cllr(p, y_test)
score_nb = liarm.cllr(p_nb, y_test)
score_clf = liarm.cllr(p_clf, y_test)
print('CLLR', score, score_nb, score_clf)'''

test = 1
