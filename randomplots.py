#!/usr/bin/env python3

import random

import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import pickle
from Function_file import *
import lir as liar
from sklearn.svm import SVC, LinearSVC
import scipy.stats as stats

#set parameters

#graphs
plt.rcParams.update({'figure.autolayout': True})
sns.set_style('whitegrid', {'axes.linewidth': 1, 'axes.edgecolor':'black'}) #0.15 is the grey variant
sns.set_palette('muted')
sns.set_context("paper", rc={"lines.linewidth": 1.5, "font.size": 18, "axes.labelsize":16, "axes.titlesize":18, "xtick.labelsize":14,
                             "ytick.labelsize":14, "legend.fontsize":16,}) #fontscale=1,5
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

hist_title = 'Histogram scores'
ECE_title = 'ECE plot'
PAV_title = 'PAV plot'
Tippet_title = 'Tippett plot'
repeat = 100
var = 1
sample_size_total = [600]
train_samples_vec = [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 350, 500, 1000, 5000]
n_freq_total = [10, 12, 17, 20, 22, 25, 35, 50, 100, 200, 300]


test = np.log2(1+1/1000000)*(1/2) + np.log2(1+1/1000000)*(1/2)


x = np.linspace(0,1,100)

fig = plt.figure(**{})
plt.plot(x, stats.norm.pdf(x, 0.6, 0.075) + 0.001, color = colors[0], label= 'same-speaker scores')
plt.plot(x, stats.norm.pdf(x, 0.37, 0.1) + 0.001, color = colors[1], label= 'different-speaker scores')
plt.plot(x, (stats.norm.pdf(x, 0.6, 0.075) + 0.201),  '--', color = colors[0])
plt.plot(x, (stats.norm.pdf(x, 0.37, 0.1)+ 0.101),  '--', color = colors[1])
#plt.plot(0.7, stats.norm.pdf(0.7, 0.6, 0.075) + 0.201, 'o', color = 'k')
#plt.plot(0.7, stats.norm.pdf(0.7, 0.37, 0.1) + 0.101, 'o', color = 'k')
#plt.vlines(0.7, 0, stats.norm.pdf(0.7, 0.6, 0.075) + 0.201)
plt.axis([0,1,0,7.2])
plt.xlabel('Score')
plt.legend()
plt.title('Score to likelihood ratio')
plt.savefig('scorebase')
plt.show()

