#!/usr/bin/env python3

import builtins
import os
from matplotlib.lines import Line2D
import string
import numpy as np
import random
import seaborn as sns
import re
import json
from nltk import word_tokenize
from tqdm import tqdm
import collections
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance
import math as m
import matplotlib.pyplot as plt
import pickle
import lir as liar

LOG = logging.getLogger(__name__)


class extract_words():
    def __init__(self, wordlist):
        self.wordlist = wordlist

    def __call__(self, texts):
        newtexts = []
        for text in texts:
            if text in self.wordlist:
                newtexts.append(text)
        return newtexts


class extract_words_new():
    def __call__(self, texts):
        yield from texts


class rearrange_samples_CGN():
    def __init__(self, n, wordlist):
        self.n = n
        self.wordlist = wordlist

    def __call__(self, items):
        allwords = [word for word in items]
        if len(allwords) < self.n:
            resampled_items = []
            print(items)
            return resampled_items
        if len(allwords) < (2*self.n):
            resampled_items = []
            return resampled_items
        else:
            nitems = len(allwords) // self.n
            itemsize = self.n
            resampled_items = []
            resampled_itemsF = []
            for i in range(nitems):
                resampled_items.append(allwords[int(i * itemsize):int(i * itemsize + itemsize)])
            LOG.info('rearrange_samples: %d elements in %d items -> %d items' % (
                len(allwords), len(items), len(resampled_items)))
        for text in resampled_items:
            newtext = []
            for word in text:
                if word in self.wordlist:
                    newtext.append(word)
            resampled_itemsF.append(newtext)
        return resampled_itemsF

class rearrange_samples_new():
    def __init__(self, n, wordlist):
        self.n = n
        self.wordlist = wordlist

    def __call__(self, items):
        for i in range(len(items) // self.n):
            yield [ word for word in items[i*self.n : (i+1)*self.n] if word in self.wordlist ]


class rearrange_samples():
    def __init__(self, n):
        self.n = n

    def __call__(self, items):
        allwords = [word for word in items]
        if len(allwords) < self.n:
            resampled_items = []
            print(items)
            return resampled_items
        else:
            nitems = len(allwords) // self.n
            itemsize = self.n
            resampled_items = []
            for i in range(nitems):
                resampled_items.append(allwords[int(i * itemsize):int(i * itemsize + itemsize)])
            LOG.debug('rearrange_samples: %d elements in %d items -> %d items' % (
                len(allwords), len(items), len(resampled_items)))
            return resampled_items


class create_feature_vector():
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, samples):
        if len(samples) == 0:
            return []
        else:
            vectorizer = TfidfVectorizer(analyzer='word', use_idf=False, norm=None, vocabulary=self.vocabulary,
                                         tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
            return vectorizer.fit_transform(samples).toarray()


def opentext(path, digest):
    with open(path, 'r') as f:
        bom = f.read(2)
    encoding = 'utf8'  # if bom == b'\xfe\xff' else 'ascii'
    return Function_file.open(path, digest, algorithm='sha256', encoding=encoding, mode='t')


def compile_data_CGN(string):
    speakers = collections.defaultdict(list)  # create empty dictionary list
    for digest, filepath in tqdm(list(read_list(string)), desc='compiling data'):  # progress bar
        speakerid = str(re.findall('N[0-9]{5}_', os.path.basename(filepath)))  # basename path
        with open(filepath, encoding='ISO-8859-1') as f:
            texts = read_session(f)
            speakers[speakerid].extend(texts)
    return speakers


def get_frequent_words(speakers, n):
    freq = collections.defaultdict(int)
    for word in speakers.values():
        for item in word:
            freq[item] += 1
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return freq[:n]


def get_frequent_ngrams(speakers, n):
    freq = collections.defaultdict(int)
    for word in speakers.values():
        samples = [' '.join(word)]

        for item in word:
            freq[item] += 1
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return freq[:n]


def get_frequent_table(speakers, wordlist, n):
    freq = collections.defaultdict(int)
    freq_speak = collections.defaultdict(int)
    sum1=0
    sumF = 0
    for word in wordlist.values():
        for item in word:
            freq[item] += 1
    for word in speakers.values():
        for item in word:
            freq_speak[item] += 1
            sum1 = sum1 + 1
    print(sum1)
    totalwords = sum(freq_speak.values())
    print(totalwords)
    freq_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]
    freq_words = list(zip(*freq_words))[0]
    freq_words_speak = sorted(freq_speak.items(), key=lambda x: x[1], reverse=True)
    freq_words_speak = list(zip(*freq_words_speak))[0]
    for key, value in sorted(freq.items(), key=lambda x: x[1], reverse=True):
        if (key in freq_words_speak and key in freq_words):
            print('\item {0:}: {1:} x - {2:}\%'.format(key, freq_speak[key], '%.2f' % ((freq_speak[key] / totalwords) * 100.0)))
            sumF = sumF + freq_speak[key]
        else:
            if key in freq_words:
                print('\item {0:}: {1:} x - {2:}\%'.format(key, value, '%.2f' % (0.00)))
    print('Total words = ' + str(sumF) + '   perc =   ' + str(sumF/sum1))
    return freq_words[:n]


def filter_texts_size(speakerdict, wordlist, aantal_woorden):
    filters = [
        extract_words(wordlist),
        rearrange_samples(aantal_woorden),
        create_feature_vector(wordlist),
    ]

    filtered = {}
    for label, texts in speakerdict.items():
        LOG.debug('filter in subset {}'.format(label))
        for f in filters:
            texts = f(texts)
        #if len(texts) == 0:
        #    print(label)
        if len(texts) != 0:
            filtered[label] = texts
    return filtered

def filter_texts_size_new(speakerdict, wordlist, aantal_woorden):
    filters = [
        extract_words_new(),
        rearrange_samples_new(aantal_woorden, wordlist),
        create_feature_vector(wordlist),
    ]
    filtered = {}
    for label, texts in speakerdict.items():
        LOG.debug('filter in subset {}'.format(label))
        for f in filters:
            texts = list(f(texts))
        #if len(texts) == 0:
        #    print(label)
        if len(texts) != 0:
            filtered[label] = texts
    return filtered

def filter_texts_size_CGN(speakerdict, wordlist, aantal_woorden):
    filters = [
        extract_words_new(),
        rearrange_samples_CGN(aantal_woorden, wordlist),
        create_feature_vector(wordlist),
    ]
    filtered = {}
    for label, texts in speakerdict.items():
        LOG.info('filter in subset {}'.format(label))
        for f in filters:
            texts = f(texts)
        #if len(texts) == 0:
        #    print(label)
        if len(texts) != 0:
            filtered[label] = texts
    return filtered


def to_vector_size(speakers):
    labels = []
    features = []
    for label, texts in speakers.items():
        speaker_id = int(re.sub('[^0-9]', '', label))
        labels.append(np.ones(len(texts)) * speaker_id)
        features.append(texts)

    return np.concatenate(features), np.concatenate(labels)


def to_vector_size_CGN(speakers, str):
    labels = []
    features = []
    distinct_labels = sorted(speakers.keys())
    for label, texts in speakers.items():
        labels.extend([re.findall('N[0-9]{5}_', label) for i in range(len(texts))])
        features.append(texts)

    return np.concatenate(features), np.ravel(np.array(labels))


def to_vector(speakers, str):
    labels = []
    features = []
    distinct_labels = sorted(speakers.keys())
    for label, texts in speakers.items():
        labels.extend(re.findall('[0-9]{3}', label))
        features.append(texts)

    return np.concatenate(features), np.array(labels)


def print_overview(speakers):
    for label, texts in speakers.items():
        print('label: {}; {} texts'.format(label, len(texts)))


def ss_feature(X, y, dist, maxsamples):
    draw_set = []
    labels_ss = []
    features_ss = []
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            if y[i] == y[j]:
                draw_set.append([i, j])
    draw_set = np.asarray(draw_set)
    trainingsample = random.sample(range(len(draw_set)), min(len(draw_set), maxsamples))
    train_set = np.asarray([draw_set[i] for i in trainingsample])
    for z in range(len(train_set)):
        if dist == 'abs':
            diff = abs(X[train_set[z, 1]] - X[train_set[z, 0]])
        if dist == 'bray':
            diff = (abs(X[train_set[z, 1]] - X[train_set[z, 0]])) / (abs(X[train_set[z, 1]] + X[train_set[z, 0]]) + 1)
        if dist == 'shan':
            diff = shanjen_vector(X[train_set[z, 1]], X[train_set[z, 0]])
        features_ss.append([diff])
        labels_ss.extend('1')
    return labels_ss, features_ss

def ds_feature(X, y, dist, maxsamples):
    draw_set = []
    labels_ds = []
    features_ds = []
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            if y[i] != y[j]:
                draw_set.append([i, j])
    draw_set = np.asarray(draw_set)
    trainingsample = random.sample(range(len(draw_set)), min(len(draw_set), maxsamples))
    train_set = np.asarray([draw_set[i] for i in trainingsample])
    for z in range(len(train_set)):
        if dist == 'abs':
            diff = abs(X[train_set[z, 1]] - X[train_set[z, 0]])
        if dist == 'bray':
            diff = (abs(X[train_set[z, 1]] - X[train_set[z, 0]])) / (abs(X[train_set[z, 1]] + X[train_set[z, 0]]) + 1)
        if dist == 'shan':
            diff = shanjen_vector(X[train_set[z, 1]], X[train_set[z, 0]])
        features_ds.append([diff])
        labels_ds.extend('0')
    return labels_ds, features_ds


def ss_score(X, y, dist, nfeat, maxsamples):
    draw_set = []
    labels_ss = []
    features_ss = []
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            if y[i] == y[j]:
                draw_set.append([i, j])
    draw_set = np.asarray(draw_set)
    lenset = len(draw_set)
    trainingsample = random.sample(range(len(draw_set)), min(len(draw_set), maxsamples))
    train_set = np.asarray([draw_set[i] for i in trainingsample])
    for z in range(len(train_set)):
        if dist == 'man':
            diff = distance.cityblock(X[train_set[z, 1]], X[train_set[z, 0]])
        if dist == 'st':
            diff = distance.cityblock(X[train_set[z, 1]], X[train_set[z, 0]]) * (1 / nfeat)
        if dist == 'euc':
            diff = distance.euclidean(X[train_set[z, 1]], X[train_set[z, 0]])
        if dist == 'cos':
            diff = distance.cosine(X[train_set[z, 1]], X[train_set[z, 0]])
        if dist == 'corr':
            diff = distance.correlation(X[train_set[z, 1]], X[train_set[z, 0]])
        if dist == 'bray':
            diff = distance.braycurtis(X[train_set[z, 1]], X[train_set[z, 0]])
        if dist == 'shan':
            diff = distance.jensenshannon(X[train_set[z, 1]], X[train_set[z, 0]])
        features_ss.append([diff])
        labels_ss.extend('1')
    return labels_ss, features_ss#, lenset


def ds_score(X, y, dist, nfeat, maxsamples):
    draw_set = []
    labels_ds = []
    features_ds = []
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            if y[i] != y[j]:
                draw_set.append([i, j])
    draw_set = np.asarray(draw_set)
    lenset = len(draw_set)
    trainingsample = random.sample(range(len(draw_set)), min(len(draw_set), maxsamples))
    train_set = np.asarray([draw_set[i] for i in trainingsample])
    for z in range(len(train_set)):
        if dist == 'man':
            diff = distance.cityblock(X[train_set[z, 1]], X[train_set[z, 0]])
        if dist == 'st':
            diff = distance.cityblock(X[train_set[z, 1]], X[train_set[z, 0]]) * (1 / nfeat)
        if dist == 'euc':
            diff = distance.euclidean(X[train_set[z, 1]], X[train_set[z, 0]])
        if dist == 'cos':
            diff = distance.cosine(X[train_set[z, 1]], X[train_set[z, 0]])
        if dist == 'corr':
            diff = distance.correlation(X[train_set[z, 1]], X[train_set[z, 0]])
        if dist == 'bray':
            diff = distance.braycurtis(X[train_set[z, 1]], X[train_set[z, 0]])
        if dist == 'shan':
            diff = distance.jensenshannon(X[train_set[z, 1]], X[train_set[z, 0]])
        features_ds.append([diff])
        labels_ds.extend('0')
    return labels_ds, features_ds#, lenset


def calc_LRfeat_ss(X, y, maxsamples):
    draw_set = []
    labels_ss = []
    features_ss = []
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            if y[i] == y[j]:
                draw_set.append([i, j])
    draw_set = np.asarray(draw_set)
    trainingsample = random.sample(range(len(draw_set)), min(len(draw_set), maxsamples))
    train_set = np.asarray([draw_set[i] for i in trainingsample])
    for z in range(len(train_set)):
        diff = np.concatenate((X[train_set[z, 1]], X[train_set[z, 0]]))
        features_ss.append([diff])
        labels_ss.extend('1')
    return features_ss, labels_ss


def calc_LRfeat_ds(X, y, maxsamples):
    draw_set = []
    labels_ds = []
    features_ds = []
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            if y[i] != y[j]:
                draw_set.append([i, j])
    draw_set = np.asarray(draw_set)
    trainingsample = random.sample(range(len(draw_set)), min(len(draw_set), maxsamples))
    train_set = np.asarray([draw_set[i] for i in trainingsample])
    for z in range(len(train_set)):
        diff = np.concatenate((X[train_set[z, 1]], X[train_set[z, 0]]))
        features_ds.append([diff])
        labels_ds.extend('0')
    return features_ds, labels_ds

def LR_acc_calc(LR, y_t):
    LR_test = LR.copy()
    LR_test = np.asarray(LR_test)
    LR_test[LR_test < 1] = '0'
    LR_test[LR_test > 1] = '1'

    count = 0
    for i in range(len(y_t)):
        if y_t[i] == LR_test[i]:
            count += 1
    accur = count / len(y_t)
    return LR_test, accur


def shanjen_vector(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p, axis=0)
    q = q / np.sum(q, axis=0)
    m = (p + q) / 2.0
    left = distance.rel_entr(p, m)
    right = distance.rel_entr(q, m)
    return np.sqrt((left + right) / 2.0)

def trainingset(y, maxsamples, key):
    draw_set = []
    if key == 'ss':
        for i in range(len(y)):
            for j in range(i+1, len(y)):
                if y[i] == y[j]:
                    draw_set.append([i, j])
    if key == 'ds':
        for i in range(len(y)):
            for j in range(i+1, len(y)):
                if y[i] != y[j]:
                    draw_set.append([i, j])
    draw_set = np.asarray(draw_set)
    trainingsample = random.sample(range(len(draw_set)), min(len(draw_set), maxsamples))
    return trainingsample

def calc_like(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq, XXX):
    product = (2 * m.pi) ** n_freq * (np.linalg.det(covar_pop)) ** (-1 / 2) * (np.linalg.det(covar_source)) ** (
                -1 / 2) * np.linalg.det((covar_source_1 + covar_pop_1)) ** (-1 / 2)
    exponent = m.exp(
        (-1 / 2) * np.transpose(np.reshape(XXX, (n_freq, 1))).dot(covar_source_1).dot(np.reshape(XXX, (n_freq, 1))) - (
                    1 / 2) * np.transpose(mean_col).dot(covar_pop_1).dot(mean_col))
    exponent2 = m.exp((1 / 2) * (np.transpose(XXX).dot(covar_source_1) + np.transpose(mean_col).dot(covar_pop_1)).dot(
        (np.linalg.inv((covar_pop_1 + covar_source_1)))).dot((covar_source_1.dot(XXX) + covar_pop_1.dot(mean_col))))
    tot = product * exponent * exponent2
    return tot


def calc_like_common(covar_pop, covar_source, covar_pop_1, covar_source_1, mean_col, n_freq, XXX):
    tot = ((2 * m.pi) ** ((-1 * n_freq)) * (np.linalg.det(covar_pop)) ** (-1) * (np.linalg.det(covar_source)) ** (
        -1) * np.linalg.det((covar_source_1 + 2 * covar_pop_1)) ** (-1 / 2)) * (m.exp((-1 / 2) * (np.transpose(np.reshape(XXX[0:n_freq], (n_freq, 1))).dot(covar_source_1).dot(
        np.reshape(XXX[0:n_freq], (n_freq, 1))) + np.transpose(np.reshape(XXX[n_freq:len(XXX)], (n_freq, 1))).dot(
        covar_source_1).dot(np.reshape(XXX[n_freq:len(XXX)], (n_freq, 1)))) - (1 / 2) * np.transpose(mean_col).dot(
        covar_pop_1).dot(mean_col))) * (m.exp((1 / 2) * ((np.transpose(XXX[0:n_freq]).dot(covar_source_1) + (
        np.transpose(XXX[n_freq:len(XXX)]).dot(covar_source_1)) + np.transpose(mean_col).dot(covar_pop_1)).dot(
        (np.linalg.inv((covar_pop_1 + 2 * covar_source_1)))).dot(
        (covar_source_1.dot((XXX[0:n_freq] + XXX[n_freq:len(XXX)])) + covar_pop_1.dot(mean_col))))))
    return tot

def plot_boxplot(Values, labels, xaxis, yaxis, boxtitle, on_screen=None, savfig=None, rotation =0, kw_figure={}):
    fig = plt.figure(**kw_figure)
    ax = fig.add_subplot(111)
    bp = ax.boxplot(Values)
    ax.set_xticklabels(labels, rotation=rotation)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    if boxtitle is not None:
        plt.title(boxtitle)
    if on_screen:
        plt.show()
    if savfig is not None:
        plt.savefig(savfig)


def plot_boxplot_line(X, Values, mean, xaxis, yaxis, boxtitle, color, legenda, on_screen=None, savfig=None, rotation=None, kw_figure={}):
    plt.figure(**kw_figure)
    ax1 = sns.boxplot(x=X, y=Values)#, width = 0.3)
    if rotation is not None:
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=rotation)
    for i, box in enumerate(ax1.artists):
        box.set_edgecolor('black')
        box.set_facecolor('white')
        box.set_linewidth(1)
        # iterate over whiskers and median lines
        for j in range(6 * i, 6 * (i + 1)):
            ax1.lines[j].set_color('black')
            ax1.lines[j].set_linewidth(1)
            if j == (6 * i + 4):
                ax1.lines[j].set_color(color[1])
    with plt.rc_context({'lines.linewidth': 1}):
        point = sns.pointplot(x=X, y=mean, color=color[0])
        plt.setp(point.collections[0], alpha=.6)  # for the markers
        for q in range(6 * (i + 1), len(point.lines)):
            plt.setp(point.lines[q], alpha=.6)
    legend_elements = [Line2D([0], [0], color='#89A8E0', lw=2)]
    plt.legend(legend_elements, [legenda])
    #points = point.collections[0]
    #size = points.get_sizes().item()
    #new_sizes = [size * 0.5]
    #points.set_sizes(new_sizes)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    if boxtitle is not None:
        plt.title(boxtitle)
    if savfig is not None:
        plt.savefig(savfig)
    if on_screen:
        plt.show()


def load_testdata(var, repeat, filename):
    print('##################################################')
    print(filename)
    print('##################################################\n')
    with open(filename, 'rb') as f:
        cllr_stat, cllr_mean, LR_acc_overall, LR_ACC_mean, labels_boxplot, LR_overall, labels_overall = pickle.load(f)
    cllr_func = []
    acc_func = []
    cllr_overall = []
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
        cllr_overall.append(cllr_mean_check)
        cllr_func.append(np.mean(cllr_mean_check))
        acc_func.append(np.mean(acc))
        print('cllr:' + str(np.mean(cllr_mean_check)))
        print('cllrmin:' + str(np.mean(cllr_mean_min)))
        print('cllrcal:' + str(np.mean(cllr_mean_cal)))
        print('Acc:' + str(np.mean(acc)))
        print('##########################################################' + str(i))
    return cllr_func, acc_func

def plot_tippetmulti(parameter, LRs, labels, par, colorset, savefig=None, show=None, titleplot=None, kw_figure={}):
    """
    plots the 10log lrs tippett plt
    """
    plt.figure(**kw_figure)
    styles = ['-','-.',':']
    for i in range(np.size(LRs)):
        LR = LRs[i]
        label = labels[i]
        xplot = np.linspace(np.min(np.log10(LR)) - 0.1, np.max(np.log10(LR)) + 0.1, 100)
        LR_0, LR_1 = liar.util.Xy_to_Xn(LR, label)
        perc0 = (sum(i > xplot for i in np.log10(LR_0)) / len(LR_0)) * 100
        perc1 = (sum(i > xplot for i in np.log10(LR_1)) / len(LR_1)) * 100
        titletip= 'LRs for '+ par + str(parameter[i])
        plt.plot(xplot, perc1, label=titletip, color = colorset[i], linestyle= styles[i])
        plt.plot(xplot, perc0, color = colorset[i], linestyle= styles[i])

    plt.axvline(x=0, color='k', linestyle='--')
    plt.xlabel('Log10 likelihood ratio')
    plt.ylabel('Cumulative proportion')
    plt.legend()
    if titleplot is not None:
        plt.title(titleplot)
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    if show or savefig is None:
        plt.show()