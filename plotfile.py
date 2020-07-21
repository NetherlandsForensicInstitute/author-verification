#!/usr/bin/env python3.7

import random

import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import pickle
from Function_file import *

from sklearn.svm import SVC, LinearSVC

#set parameters

#graphs
plt.rcParams.update({'figure.autolayout': True})
sns.set_style('whitegrid', {'axes.linewidth': 1, 'axes.edgecolor':'black'}) #0.15 is the grey variant
sns.set_palette('muted')
sns.set_context("paper", rc={"lines.linewidth": 1.5, "font.size": 18, "axes.labelsize":16, "axes.titlesize":18, "xtick.labelsize":14,
                             "ytick.labelsize":14, "legend.fontsize":16}) #fontscale=1,5
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

hist_title = 'Histogram scores'
ECE_title = 'ECE plot'
PAV_title = 'PAV plot'
Tippet_title = 'Tippett plot'
repeat = 100
var = 5
spec_samples_vec = [2, 4, 6, 8, 10]
sample_size_total = [250, 375, 500, 625, 750]
train_samples_vec = [10, 50, 75, 100, 200, 300, 500, 1000, 2000, 3000, 4000]
n_freq_total = [50, 100, 150, 200, 250, 300, 400, 600]

'''

filename = 'sLRcommonAt10Atr[170]rep100ss[250, 375, 500, 625, 750]F[20]'
scllr_15, sacc_15 = load_testdata(var, repeat, filename)
filename = 'sLRcommonAt10Atr[170]rep100ss[250, 375, 500, 625, 750]F[35]'
scllr_25, sacc_25 = load_testdata(var, repeat, filename)
filename = 'sLRcommonAt10Atr[170]rep100ss[250, 375, 500, 625, 750]F[50]'
scllr_30, sacc_30 = load_testdata(var, repeat, filename)

fig = plt.figure(**{})
plt.plot(sample_size_total, scllr_15, color= colors[0], label = '$\mathregular{F_{\#}=20}$')
plt.plot(sample_size_total, scllr_25, color= colors[1], label = '$\mathregular{F_{\#}=35}$')
plt.plot(sample_size_total, scllr_30,  color= colors[2], label = '$\mathregular{F_{\#}=50}$')
plt.xlabel('Samplesize (N)')
plt.ylabel('Cllr')
plt.title('Cllr LR score common source')
plt.legend()
plt.savefig('plot_cllr_slrcommon')
plt.show()

fig = plt.figure(**{})
plt.plot(sample_size_total, sacc_15, color= colors[0], label = '$\mathregular{F_{\#}=20}$')
plt.plot(sample_size_total, sacc_25, color= colors[1], label = '$\mathregular{F_{\#}=35}$')
plt.plot(sample_size_total, sacc_30,  color= colors[2], label = '$\mathregular{F_{\#}=50}$')
plt.xlabel('Samplesize (N)')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy LR score common source')
plt.savefig('plot_Acc_slrcommon')
plt.show()


test = 'test'
filename = 'sLRspecAt10Atr[170]rep100ss[250]F[20]spec[2, 4, 6, 8, 10]'
scllr_15, sacc_15 = load_testdata(var, repeat, filename)
filename = 'sLRspecAt10Atr[170]rep100ss[250]F[50]spec[2, 4, 6, 8, 10]'
scllr_25, sacc_25 = load_testdata(var, repeat, filename)
filename = 'sLRspecAt10Atr[170]rep100ss[250]F[75]spec[2, 4, 6, 8, 10]'
scllr_30, sacc_30 = load_testdata(var, repeat, filename)

fig = plt.figure(**{})
plt.plot(spec_samples_vec, scllr_15, color= colors[0], label = '$\mathregular{F_{\#}=20}$')
plt.plot(spec_samples_vec, scllr_25, color= colors[1], label = '$\mathregular{F_{\#}=50}$')
plt.plot(spec_samples_vec, scllr_30,  color= colors[2], label = '$\mathregular{F_{\#}=75}$')
plt.xlabel('Number of specific speaker samples ' +  '$\mathregular{S_{\# spec}}$')
plt.ylabel('Cllr')
plt.title('Cllr LR score specific source')
plt.legend()
plt.savefig('plot_cllr_slrspec')
plt.show()

fig = plt.figure(**{})
plt.plot(spec_samples_vec, sacc_15, color= colors[0], label = '$\mathregular{F_{\#}=20}$')
plt.plot(spec_samples_vec, sacc_25, color= colors[1], label = '$\mathregular{F_{\#}=50}$')
plt.plot(spec_samples_vec, sacc_30,  color= colors[2], label = '$\mathregular{F_{\#}=75}$')
plt.xlabel('Number of specific speaker samples ' +  '$\mathregular{S_{\# spec}}$')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy LR score specific source')
plt.savefig('plot_Acc_slrspec')
plt.show()


filename = 'DistshanOWNAt10Atr[190]rep100ss[250, 500, 750, 1000, 1250, 1500, 2000]F[50]'
cllr_10, acc_10 = load_testdata(var, repeat, filename)
filename = 'DistshanOWNAt10Atr[190]rep100ss[250, 500, 750, 1000, 1250, 1500, 2000]F[100]'
cllr_15, acc_15 = load_testdata(var, repeat, filename)
filename = 'DistshanOWNAt10Atr[190]rep100ss[250, 500, 750, 1000, 1250, 1500, 2000]F[200]'
cllr_25, acc_25 = load_testdata(var, repeat, filename)
filename = 'DistshanOWNAt10Atr[190]rep100ss[250, 500, 750, 1000, 1250, 1500, 2000]F[300]'
cllr_50, acc_50 = load_testdata(var, repeat, filename)
filename = 'DistshanOWNAt10Atr[190]rep100ss[250, 500, 750, 1000, 1250, 1500, 2000]F[400]'
cllr_75, acc_75 = load_testdata(var, repeat, filename)
filename = 'DistshanOWNAt10Atr[190]rep100ss[250, 500, 750, 1000, 1250, 1500, 2000]F[500]'
cllr_100, acc_100 = load_testdata(var, repeat, filename)
filename = 'DistshanOWNAt10Atr[190]rep100ss[250, 500, 750, 1000, 1250, 1500, 2000]F[600]'
cllr_150, acc_150 = load_testdata(var, repeat, filename)


fig = plt.figure(**{})
plt.plot(sample_size_total, cllr_10, color= colors[0], label = '$\mathregular{F_{\#}=50}$')
plt.plot(sample_size_total, cllr_15, color= colors[1], label = '$\mathregular{F_{\#}=100}$')
plt.plot(sample_size_total, cllr_25, color= colors[2], label = '$\mathregular{F_{\#}=200}$')
#plt.plot(sample_size_total, cllr_50,  color= colors[3], label = '$\mathregular{F_{\#}=300}$')
plt.plot(sample_size_total, cllr_75,  color= colors[4], label = '$\mathregular{F_{\#}=400}$')
#plt.plot(sample_size_total, cllr_100,  color= colors[5], label = '$\mathregular{F_{\#}=500}$')
plt.plot(sample_size_total, cllr_150,  color= colors[6], label = '$\mathregular{F_{\#}=600}$')
plt.xlabel('Sample size (N)')
plt.ylabel('Cllr')
plt.title('Cllr distance')
plt.legend()
plt.savefig('plot_cllr_distance')
plt.show()


fig = plt.figure(**{})
plt.plot(sample_size_total, acc_10, color= colors[0], label = '$\mathregular{F_{\#}=50}$')
plt.plot(sample_size_total, acc_15, color= colors[1], label = '$\mathregular{F_{\#}=100}$')
plt.plot(sample_size_total, acc_25, color= colors[2], label = '$\mathregular{F_{\#}=200}$')
#plt.plot(sample_size_total, acc_50,  color= colors[3], label = '$\mathregular{F_{\#}=300}$')
plt.plot(sample_size_total, acc_75,  color= colors[4], label = '$\mathregular{F_{\#}=400}$')
#plt.plot(sample_size_total, acc_100,  color= colors[5], label = '$\mathregular{F_{\#}=500}$')
plt.plot(sample_size_total, acc_150,  color= colors[6], label = '$\mathregular{F_{\#}=600}$')
plt.xlabel('Sample size (N)')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy distance')
plt.savefig('plot_Acc_distance')
plt.show()
var=6
sample_size_total = [250, 500, 750, 1000, 1250, 1500]
filename = 'XGBAt10Atr[190]rep100ss[250, 500, 750, 1000, 1250, 1500]F[50]'
cllr_10, acc_10 = load_testdata(var, repeat, filename)
filename = 'XGBAt10Atr[190]rep100ss[250, 500, 750, 1000, 1250, 1500]F[100]'
cllr_15, acc_15 = load_testdata(var, repeat, filename)
#filename = 'SVMAt10Atr[190]rep100ss[250, 500, 750, 1000, 1500]F[150]'
#cllr_25, acc_25 = load_testdata(var, repeat, filename)
filename = 'XGBAt10Atr[190]rep100ss[250, 500, 750, 1000, 1250, 1500]F[200]'
cllr_50, acc_50 = load_testdata(var, repeat, filename)
#filename = 'SVMAt10Atr[190]rep100ss[250, 500, 750, 1000, 1500]F[250]'
#cllr_75, acc_75 = load_testdata(var, repeat, filename)
filename = 'XGBAt10Atr[190]rep100ss[250, 500, 750, 1000, 1250, 1500]F[300]'
cllr_100, acc_100 = load_testdata(var, repeat, filename)
filename = 'XGBAt10Atr[190]rep100ss[250, 500, 750, 1000, 1250, 1500]F[400]'
cllr_150, acc_150 = load_testdata(var, repeat, filename)


fig = plt.figure(**{})
plt.plot(sample_size_total, cllr_10, color= colors[8], label = '$\mathregular{F_{\#}=50}$')
plt.plot(sample_size_total, cllr_15, color= colors[0], label = '$\mathregular{F_{\#}=100}$')
#plt.plot(sample_size_total, cllr_25, color= colors[2], label = '$\mathregular{F_{\#}=200}$')
plt.plot(sample_size_total, cllr_50,  color= colors[1], label = '$\mathregular{F_{\#}=200}$')
#plt.plot(sample_size_total, cllr_75,  color= colors[4], label = '$\mathregular{F_{\#}=400}$')
plt.plot(sample_size_total, cllr_100,  color= colors[2], label = '$\mathregular{F_{\#}=300}$')
plt.plot(sample_size_total, cllr_150,  color= colors[3], label = '$\mathregular{F_{\#}=400}$')
plt.xlabel('Sample length (N)')
plt.ylabel('Cllr')
plt.title('Cllr  XGBoost')
plt.legend()
plt.savefig('plot_cllr_XGB')
plt.show()


fig = plt.figure(**{})
plt.plot(sample_size_total, acc_10, color= colors[8], label = '$\mathregular{F_{\#}=50}$')
plt.plot(sample_size_total, acc_15, color= colors[0], label = '$\mathregular{F_{\#}=100}$')
#plt.plot(sample_size_total, acc_25, color= colors[2], label = '$\mathregular{F_{\#}=200}$')
plt.plot(sample_size_total, acc_50,  color= colors[1], label = '$\mathregular{F_{\#}=200}$')
#plt.plot(sample_size_total, acc_75,  color= colors[4], label = '$\mathregular{F_{\#}=400}$')
plt.plot(sample_size_total, acc_100,  color= colors[2], label = '$\mathregular{F_{\#}=300}$')
plt.plot(sample_size_total, acc_150,  color= colors[3], label = '$\mathregular{F_{\#}=400}$')
plt.xlabel('Sample length (N)')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy XGBoost')
plt.savefig('plot_Acc_XGB')
plt.show()'''

with open('DistshanOWNAt10Atr[190]rep100ss[250, 500, 750, 1000, 1250, 1500, 2000]F[200]', 'rb') as f:
    cllr_stat, cllr_mean, LR_acc_overall, LR_ACC_mean, labels_boxplot, LR_overall, labels_overall = pickle.load(f)

train_samples_vec = [10, 50, 75, 100, 200, 300, 500, 1000, 2000, 3000, 4000]
sample_size_total = [250, 500, 750, 1000, 1250, 1500]
n_freq_total = [25, 50, 100, 150, 200, 250, 300, 400, 600]
spec_samples_vec = [2, 4, 6, 8, 10]
cllr_mean_std = []
acc_mean_std = []
cllr_overall = []
cllr_mean_double = []
acc_mean_double = []
var = 6

print('##########################################################' + str('single file'))
for i in range(var):
    cllr_mean_check = []
    cllr_mean_min = []
    cllr_mean_cal = []
    cllr_mean_0 = []
    cllr_mean_1 = []
    acc = []
    for z in range(repeat):
        acc.append(LR_acc_overall[i][z])
        cllr_mean_check.append(cllr_stat[i*repeat + z].cllr)
        cllr_mean_min.append(cllr_stat[i*repeat + z].cllr_min)
        cllr_mean_cal.append(cllr_stat[i * repeat + z].cllr_cal)
        cllr_mean_0.append(cllr_stat[i * repeat + z].cllr_class0)
        cllr_mean_1.append(cllr_stat[i * repeat + z].cllr_class1)
    #fig = plt.figure(**{})
    #plt.hist(acc, bins=15)
    #plt.show()
    cllr_overall.append(cllr_mean_check)
    print('cllr:' + str(np.mean(cllr_mean_check)))
    print('cllrmin:' + str(np.mean(cllr_mean_min)))
    print('cllrcal:' + str(np.mean(cllr_mean_cal)))
    print('cllr0:' + str(np.mean(cllr_mean_0)))
    print('cllr1:' + str(np.mean(cllr_mean_1)))
    print('Acc:' + str(np.mean(acc)))
    print('##########################################################' + str(i))
    cllr_mean_double.append(np.mean(cllr_mean_check))
    acc_mean_double.append(np.mean(acc))
'''
fig = plt.figure(**{})
plt.plot(spec_samples_vec, acc_mean_double, color= colors[0], label = '$\mathregular{F_{\#}=30,N=200}$')
plt.xlabel('Number of specific speaker samples $\mathregular{S_{\# spec}}$')
plt.ylabel('Accuracy [%]')
plt.legend()
plt.title('Accuracy LR score specific source')
plt.savefig('plot_Acc_slrscom')
plt.show()

fig = plt.figure(**{})
plt.plot(spec_samples_vec, cllr_mean_double,  color= colors[1], label = '$\mathregular{F_{\# }=30,N=200}$')
plt.xlabel('Number of specific speaker samples $\mathregular{S_{\# spec}}$')
plt.ylabel('Cllr')
plt.legend()
plt.title('Cllr LR score specific source')
plt.savefig('plot_cllr_slrcom')
plt.show()'''

plot_boxplot_line(X=sample_size_total, Values=cllr_overall, mean=cllr_mean_double,  xaxis='Sample length (N)', legenda= 'Mean Cllr', yaxis='Cllr', boxtitle='Cllr distance', savfig='boxplot_line_cllr_ss', rotation=0, on_screen=True, color = colors )
plot_boxplot_line(X=sample_size_total, Values=LR_acc_overall[0:var], mean=acc_mean_double,  xaxis='Sample length (N)', legenda= 'Mean accuracy', yaxis='Accuracy', boxtitle='Accuracy distance', savfig='boxplot_line_acc_ss', rotation=0, on_screen=True, color = colors )


#plot_boxplot_line(X=n_freq_total, Values=cllr_overall, mean=cllr_mean_double,  xaxis='Number of frequent words $\mathregular{F_{\#}}$', legenda= 'Mean Cllr', yaxis='Cllr', boxtitle='Cllr distance CGN', savfig='boxplot_line_cllr_Fcgn', rotation=0, on_screen=True, color = colors )
#plot_boxplot_line(X=n_freq_total, Values=LR_acc_overall, mean=acc_mean_double,  xaxis='Number of frequent words $\mathregular{F_{\#}}$', legenda= 'Mean accuracy', yaxis='Accuracy', boxtitle='Accuracy distance CGN', savfig='boxplot_line_acc_Fcgn', rotation=0, on_screen=True, color = colors )

#plot_boxplot_line(X=train_samples_vec, Values=cllr_overall, mean=cllr_mean_double,  xaxis='Number of samples $\mathregular{S_{\#}}$', legenda= 'Mean Cllr', yaxis='Cllr', boxtitle='Cllr distance', savfig='boxplot_line_cllr_S', rotation=0, on_screen=True, color = colors )
#plot_boxplot_line(X=train_samples_vec, Values=LR_acc_overall, mean=acc_mean_double,  xaxis='Number of samples $\mathregular{S_{\#}}$', legenda= 'Mean accuracy', yaxis='Accuracy', boxtitle='Accuracy distance', savfig='boxplot_line_acc_S', rotation=0, on_screen=True, color = colors )

#plot_boxplot_line(X=['$\mathregular{F_{\#}=40}$, $\mathregular{N=750}$'], Values=cllr_overall, mean=cllr_mean_double,  xaxis='Parameters', legenda= 'Mean Cllr', yaxis='Cllr', boxtitle='Cllr common source LR score', savfig='boxplot_line_cllr_bestcom', rotation=0, on_screen=True, color = colors )
#plot_boxplot_line(X=['$\mathregular{F_{\#}=40}$, $\mathregular{N=750}$'], Values=LR_acc_overall, mean=acc_mean_double,  xaxis='Parameters', legenda= 'Mean accuracy', yaxis='Accuracy', boxtitle='Accuracy common source LR score', savfig='boxplot_line_acc_bestcom', rotation=0, on_screen=True, color = colors )

#plot_boxplot_line(X=spec_samples_vec, Values=cllr_overall, mean=cllr_mean_double,  xaxis='Number of samples $\mathregular{S_{\# spec}}$', legenda= 'Mean Cllr', yaxis='Cllr', boxtitle='Cllr specific source LR score', savfig='boxplot_line_cllr_Sspec', rotation=0, on_screen=True, color = colors )
#plot_boxplot_line(X=spec_samples_vec, Values=LR_acc_overall, mean=acc_mean_double,  xaxis='Number of samples $\mathregular{S_{\# spec}}$', legenda= 'Mean accuracy', yaxis='Accuracy', boxtitle='Accuracy specific source LR score', savfig='boxplot_line_acc_Sspec', rotation=0, on_screen=True, color = colors )


'''fig = plt.figure(**{})
plt.plot(sample_size_total, cllr_mean, color= colors[0], label = 'Cllr')
plt.errorbar(sample_size_total, cllr_mean, label='95% conf', yerr=(np.asarray(cllr_mean_std)*1.96), elinewidth=1, color=colors[1], fmt='o', capsize=4)
plt.xlabel('Samplesize (N)')
plt.ylabel('Cllr')
plt.title('Distance Cllr')
plt.legend()
plt.savefig('plot_cllr_distance')
plt.show()

fig = plt.figure(**{})
plt.plot(sample_size_total, LR_ACC_mean, label='Accuracy', color= colors[0])
plt.errorbar(sample_size_total, LR_ACC_mean, label='95% conf', yerr=(np.asarray(acc_mean_std)*1.96), elinewidth=1, color=colors[1], fmt='o', capsize=4)
plt.xlabel('Samplesize (N)')
plt.ylabel('Accuracy [%]')
plt.legend()
plt.title('Distance accuracy')
plt.savefig('plot_Acc_distance')
plt.show()'''