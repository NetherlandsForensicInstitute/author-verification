import lir
import json

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
import seaborn as sns

from sklearn.metrics import roc_curve, confusion_matrix

main_path = 'frida/predictions'

runs = ['exp1', 'exp2', 'exp3']
specific_run = runs[1]
# prediction_file = 'predictions_per_repeat.json'
prediction_file = 'predictions_per_repeat_' + specific_run + '.json'
param_file = 'param_' + specific_run + '.json'


# for most freq words
def most_frequent(df):
    num = []
    perc = []
    mfw = 100 * (df.cumsum() / df.sum())
    mfw = mfw.round()

    for i in range(10, 110, 10):
        mfw_subset = mfw[mfw['freq'] <= i]
        num.append(mfw_subset.shape[0])

    for j in np.array([10, 50, 100, 150, 200, 300, 400, 500, 600, 1000]):
        mfw_subset = mfw.head(j)
        perc.append(mfw_subset.iat[-1, 0])

    return num, perc


# equal error rate
def eer(lrs, y):
    fpr, tpr, threshold = roc_curve(list(y), list(lrs), pos_label=1)
    fnr = 1 - tpr
    # return fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    try:
        return fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    except ValueError:
        return np.nan


# cllr_min
def cllr_min(lrs, y):
    lrs = np.array(lrs)
    y = np.array(y)
    return lir.metrics.cllr_min(lrs, y)


# cllr
def cllr(lrs, y):
    return lir.metrics.cllr(lrs, y)


# recall (true positive rate)
def recall(lrs, y):
    return np.mean(lrs[y == 1] > 1)


# precision
def precision(lrs, y):
    return np.mean(y[lrs > 1] == 1)


# percentiles
def percentiles_for_df(name, arr):
    log_arr = np.log10(arr)
    temp = np.percentile(log_arr, range(0, 101, 25))
    temp = [round(i, 4) if i < 1 else round(i, 1) for i in temp]
    return [name] + temp


def lr_histogram(lrs, y, bins=20, title=''):
    """
    plots the 10log lrs under the two hypotheses
    """
    log_lrs = np.log10(lrs)

    bins = np.histogram_bin_edges(log_lrs, bins=bins)
    points0, points1 = lir.util.Xy_to_Xn(log_lrs, y)
    fig, ax = plt.subplots()
    ax.hist(points0, bins=bins, alpha=.25, density=True)
    ax.hist(points1, bins=bins, alpha=.25, density=True)
    ax.set_xlabel('10log likelihood ratio')
    ax.set_ylabel('count')
    ax.set_title(title)

    return fig


def metrics_per_category(cols_to_group, lrs_col, y, df):
    """
    parameters
    :param cols_to_group = list of column names
    :param lrs_col = column name of the lrs
    :param y = column name of the ground truth (binary 0/1)
    :param df = dataframe with the mentioned columns

    it returns a df where each comb of values in cols_to_group the counts and the values of diff metrics
    """
    # counts
    cols = ['repeat'] + cols_to_group
    grouped_df = df.groupby(cols)[y].agg([lambda x: sum(x == 0), lambda x: sum(x)]).reset_index()
    grouped_df.rename(columns={'<lambda_0>': "total_ds", '<lambda_1>': "total_ss"}, inplace=True)

    cllr_df = predictions.groupby(cols)[['y', lrs_col]].apply(
        lambda x: cllr(x[lrs_col], x['y'])).to_frame().reset_index()
    cllr_df.rename(columns={0: "cllr"}, inplace=True)

    eer_df = predictions.groupby(cols)[['y', lrs_col]].apply(lambda x: eer(x[lrs_col], x['y'])).to_frame().reset_index()
    eer_df.rename(columns={0: "eer"}, inplace=True)

    grouped_all = grouped_df.merge(cllr_df, on=cols)
    grouped_all = grouped_all.merge(eer_df, on=cols)
    grouped_all = grouped_all.groupby(cols_to_group).agg('mean').reset_index()

    grouped_all['total_ds'] = round(grouped_all['total_ds'], 1)
    grouped_all['total_ss'] = round(grouped_all['total_ss'], 1)
    grouped_all['cllr'] = round(grouped_all['cllr'], 3)
    grouped_all['eer'] = round(grouped_all['eer'], 3)

    return grouped_all


def metrics_for_plotting(cols_to_group, df):
    """
    todo: update
    """
    # counts
    cols = ['repeat'] + cols_to_group
    final_df = pd.DataFrame()
    for col in ['lrs_voc', 'lrs_multi', 'lrs_comb_a', 'lrs_comb_b', 'lrs_feat', 'lrs_biva']:
        cllr_df = df.groupby(cols)[['y', col]].apply(lambda x: cllr(x[col], x['y'])).to_frame().reset_index()
        cllr_df.rename(columns={0: "value"}, inplace=True)
        cllr_df = cllr_df.groupby(cols_to_group).agg('mean').reset_index()
        cllr_df['metric'] = 'cllr'
        cllr_df['method'] = col.replace('lrs_', '')
        cllr_df = cllr_df[['method'] + cols_to_group + ['metric', 'value']]
        final_df = final_df.append(cllr_df, ignore_index=True)

        eer_df = df.groupby(cols)[['y', col]].apply(lambda x: eer(x[col], x['y'])).to_frame().reset_index()
        eer_df.rename(columns={0: "value"}, inplace=True)
        eer_df = eer_df.groupby(cols_to_group).agg('mean').reset_index()
        eer_df['metric'] = 'eer'
        eer_df['method'] = col.replace('lrs_', '')
        eer_df = eer_df[['method'] + cols_to_group + ['metric', 'value']]
        final_df = final_df.append(eer_df, ignore_index=True)

    return final_df

if __name__ == '__main__':

    # load and prep data

    # words and their frequencies (based on whole dataset)
    words = pd.read_json(f'{main_path}/word_frequencies.json', orient="index")
    words.rename(columns={0: "freq"}, inplace=True)
    words.sort_values(by='freq', ascending=False, inplace=True)
    top_200 = words.index[:200].tolist()

    # wc = WordCloud(width=2500, height=400)
    # wc.generate_from_frequencies(frequencies=words.to_dict()['freq'])

    # num_mfw, perc_mfw = most_frequent(words)

    # length of the conversations (excl. words that are non-existing, incomplete, distorted or unclear)
    conv_len = pd.read_json(f'{main_path}/conversation_length.json', orient="index")
    conv_len.reset_index(inplace=True)
    conv_len.rename(columns={'index': 'conv_id', 0: "num_words"}, inplace=True)
    bins_words = [24, 400, 600, 1286]
    names_words = ['n<=400', '400<n<=600', 'n>600']
    conv_len['num_words_range'] = pd.cut(conv_len['num_words'], bins_words, labels=names_words).astype('O')

    spks = list(set([conv_id[:(len(conv_id) - 2)] for conv_id in conv_len.conv_id]))

    # words and their freq per conversation for the top 200 words based on the total
    conv_rel_freq = pd.read_json(f'{main_path}/conversation_relative_frequencies.json', orient="index")
    conv_rel_freq.columns = top_200

    # speaker metadata
    spk_metadata = pd.read_csv(f'{main_path}/metadata.csv')
    spk_metadata.drop(['PartnerID', 'RoomID', 'Nationality', 'Place of birth', 'Living place(0-7yrs)',
                       'Education Background ', 'dialect', 'Group'], axis=1, inplace=True)
    spk_metadata.columns = ['spk_id', 'age', 'birth_place_father', 'birth_place_mother', 'second_language']
    spk_metadata['spk_id'] = spk_metadata['spk_id'].str.replace('Speaker', 'SP', regex=False)

    spks_in_meta = spk_metadata.spk_id.tolist()
    spks_not_in_meta = list(set(spks) - set(spks_in_meta))
    spks_in_meta_with_no_trans = list(set(spks_in_meta) - set(spks))

    spk_metadata = spk_metadata[spk_metadata['spk_id'].isin(spks)]
    spk_metadata['age'] = spk_metadata['age'].astype('Int64')

    bins = [17, 20, 30, 56]
    names = ['18-20', '21-30', '31-55']
    spk_metadata['age_range'] = pd.cut(spk_metadata['age'], bins, labels=names).astype('O')

    spk_metadata.drop(['age', 'birth_place_father', 'birth_place_mother'], axis=1, inplace=True)
    # spk_metadata['birth_place_father'] = spk_metadata['birth_place_father']. \
    #     str.replace('[mM]oroc[coan]*', 'Morocco'). \
    #     str.replace('Turk[eyish]*', 'Turkey'). \
    #     str.replace('Neatherlands', 'Netherlands', regex=False). \
    #     str.replace('Holland', 'Netherlands', regex=False). \
    #     str.replace('Dutch', 'Netherlands', regex=False). \
    #     str.replace('Amsterdam', 'Netherlands', regex=False)
    # spk_metadata['birth_place_mother'] = spk_metadata['birth_place_mother']. \
    #     str.replace('[mM]oroc[coan]*', 'Morocco'). \
    #     str.replace('Tu[r]{0,1}k[eyish]*', 'Turkey'). \
    #     str.replace('Neatherlands', 'Netherlands', regex=False). \
    #     str.replace('Hol[la][la]n[d]{0,1}', 'Netherlands'). \
    #     str.replace('Dutch', 'Netherlands', regex=False)
    spk_metadata['second_language'] = spk_metadata['second_language']. \
        str.replace('[mM]o[rc]oc[coan]*', 'Moroccan'). \
        str.replace('Tu[r]{0,1}k[eyish]*', 'Turkish'). \
        str.replace('Enligsh', 'English', regex=False)

    # # create conversation metadata
    # conv_metadata = pd.DataFrame({'session': [2, 4, 6, 8], 'location': ['indoor', 'indoor', 'outdoor', 'outdoor'],
    #                               'environment': ['quiet', 'noisy', 'quiet', 'noisy']})

    # parameters
    with open(f'{main_path}/{param_file}') as p:
        param = json.load(p)

    # predictions
    with open(f'{main_path}/{prediction_file}') as f:
        per_repeat = json.load(f)

    predictions = pd.DataFrame()
    num_pairs = pd.DataFrame()

    for key in per_repeat.keys():
        train_same, train_diff = sum(np.array(per_repeat[key]['train']) == 1), \
                                 sum(np.array(per_repeat[key]['train']) == 0)
        test_same, test_diff = sum(np.array(per_repeat[key]['y']) == 1), sum(np.array(per_repeat[key]['y']) == 0)
        temp_num_df = pd.DataFrame({'repeat': key, 'train_same': train_same, 'train_diff': train_diff,
                                    'train_all': len(per_repeat[key]['train']), 'test_same': test_same,
                                    'test_diff': test_diff, 'test_all': len(per_repeat[key]['y'])}, index=[0])

        # pairs are in the form of SPXXXXXSPXXXXX
        conv_A = ['SP' + pair.split('SP')[1] for pair in per_repeat[key]['pairs']]
        spk_A = [conv_id[:-2] for conv_id in conv_A]
        conv_B = ['SP' + pair.split('SP')[2] for pair in per_repeat[key]['pairs']]
        spk_B = [conv_id[:-2] for conv_id in conv_B]
        lrs_multi = [a * b for a, b in zip(per_repeat[key]['lrs_mfw'], per_repeat[key]['lrs_voc'])]

        temp_df = pd.DataFrame({'repeat': key, 'spk_A': spk_A, 'spk_B': spk_B, 'conv_A': conv_A, 'conv_B': conv_B,
                                'y': per_repeat[key]['y'], 'lrs_mfw': per_repeat[key]['lrs_mfw'],
                                'lrs_voc': per_repeat[key]['lrs_voc'], 'lrs_multi': lrs_multi,
                                'lrs_comb_a': per_repeat[key]['lrs_comb_a'],
                                'lrs_comb_b': per_repeat[key]['lrs_comb_b'],
                                'lrs_feat': per_repeat[key]['lrs_feat'], 'lrs_biva': per_repeat[key]['lrs_biva']})
        temp_df['y'] = temp_df['y'].astype('Int64')
        predictions = predictions.append(temp_df, ignore_index=True)
        num_pairs = num_pairs.append(temp_num_df, ignore_index=True)

    # performance results on a high level and per repeat
    metric_functions = {'cllr': cllr, 'eer': eer, 'cllr_min': cllr_min,
                        'recall': recall, 'precision': precision}
    metrics_per_repeat = pd.DataFrame()

    for col in ['lrs_mfw', 'lrs_voc', 'lrs_multi', 'lrs_comb_a', 'lrs_comb_b', 'lrs_feat', 'lrs_biva']:

        method = col.replace('lrs_', '')

        with lir.plotting.savefig(f'{main_path}/pav_{col}.png') as ax:
            ax.pav(predictions[col].array, predictions['y'].array)

        with lir.plotting.savefig(f'{main_path}/tippett_{col}.png') as ax:
            ax.tippett(predictions[col].array, predictions['y'].array)

        # calculate metrics per repeat
        for m, f in metric_functions.items():
            temp = predictions.groupby('repeat')[['y', col]].apply(lambda x: f(x[col], x['y'])).to_frame()
            temp.reset_index(level=0, inplace=True)
            temp['metric'] = m
            temp['method'] = method
            temp.rename(columns={0: 'value'}, inplace=True)
            temp = temp[['method', 'repeat', 'metric', 'value']]
            metrics_per_repeat = metrics_per_repeat.append(temp, ignore_index=True)

        # calculate avg and std of the metrics
        res_df = metrics_per_repeat.groupby(['method', 'metric']).agg(avg=('value', 'mean'),
                                                                      std=('value', 'std')).reset_index()
        res_df = res_df.pivot_table(index=['method'], columns='metric', values=['avg', 'std'])

        res_df = res_df.sort_index(axis=1, level=1)
        res_df.columns = [f'{y}_{x}' for x, y in res_df.columns]
        res_df = res_df.reindex(['mfw', 'voc', 'multi', 'comb_a', 'comb_b', 'feat', 'biva'])
        res_df = res_df.reset_index()

    # add info on predictions
    predictions['location_A'] = np.where(predictions['conv_A'].str.endswith(('2', '4')), 'indoor', 'outdoor')
    predictions['environment_A'] = np.where(predictions['conv_A'].str.endswith(('2', '6')), 'quiet', 'noisy')
    predictions['location_B'] = np.where(predictions['conv_B'].str.endswith(('2', '4')), 'indoor', 'outdoor')
    predictions['environment_B'] = np.where(predictions['conv_B'].str.endswith(('2', '6')), 'quiet', 'noisy')
    predictions = predictions.merge(conv_len.add_suffix('_A'), left_on='conv_A', right_on='conv_id_A')
    predictions = predictions.merge(conv_len.add_suffix('_B'), left_on='conv_B', right_on='conv_id_B')
    predictions = predictions.merge(spk_metadata.add_suffix('_A'), left_on='spk_A', right_on='spk_id_A')
    predictions = predictions.merge(spk_metadata.add_suffix('_B'), left_on='spk_B', right_on='spk_id_B')
    predictions['spk_locations'] = np.where(predictions['location_A'] == predictions['location_B'],
                                            predictions['location_A'], 'indoor vs outdoor')
    predictions['spk_environments'] = np.where(predictions['environment_A'] == predictions['environment_B'],
                                               predictions['environment_A'], 'quiet vs noisy')
    predictions['spk_num_words'] = np.where(predictions['num_words_range_A'] == predictions['num_words_range_B'],
                                            predictions['num_words_range_A'],
                                            np.where(((predictions['num_words_range_A'] == 'n<=400') & (
                                                        predictions['num_words_range_B'] == '400<n<=600')) |
                                                     ((predictions['num_words_range_B'] == 'n<=400') & (
                                                                 predictions['num_words_range_A'] == '400<n<=600')),
                                                     'n<=400 vs 400<n<=600',
                                                     np.where(((predictions['num_words_range_A'] == 'n<=400') & (
                                                                 predictions['num_words_range_B'] == 'n>600')) | \
                                                              ((predictions['num_words_range_B'] == 'n<=400') & (
                                                                          predictions['num_words_range_A'] == 'n>600')),
                                                              'n<=400 vs n>600', '400<n<=600 vs n>600')))
    predictions.drop(['spk_id_A', 'spk_id_B', 'conv_id_A', 'conv_id_B'], axis=1, inplace=True)

# ----------------------------------------------------------------------------------------------------------------------
st.set_page_config(layout="wide")

st.title('Fusion of a speaker and a author verification system - Results exploration')
st.markdown("""---""")

# ----------------------------------------------------------------------------------------------------------------------
st.header('FRIDA dataset')
st.write('Based on the transcriptions, we are working with ', len(spks), ' speakers. In the metadata file, there are ',
         len(spks_in_meta), ' speakers with ', len(spks_in_meta_with_no_trans), ' of them having no transcriptions',
         'Additionally, we have ', len(spks_not_in_meta), ' speakers with transcriptions but with no record in the'
                                                          ' metadata file.')

# ---------
st.subheader('Speakers demographics')
st.write('The speakers were males who were born and raised in Amsterdam.')
cols = st.columns(4)
cols[0].write(spk_metadata.age_range.value_counts(ascending=False))
cols[1].write(spk_metadata.second_language.value_counts(ascending=False))
# cols[2].write(spk_metadata.birth_place_father.value_counts(ascending=False))
# cols[3].write(spk_metadata.birth_place_mother.value_counts(ascending=False))

# ---------
# st.subheader('Sessions')
# st.write('We are focusing on the sessions 2, 4, 6, and 8, as those are the sessions for which a transcription is '
#          'available. The table belows shows were the conversation took place and under which background conditions.')
# st.write(conv_metadata)

# ---------
st.subheader('Number of words per conversation')
st.write('We have exclude all the non-existing, incomplete, distorted or unclear words.')

cols = st.columns(2)
fig1, ax1 = plt.subplots()
ax1.hist(conv_len.num_words, bins=20)
ax1.set_ylabel('counts')
ax1.set_xlabel('# of tokens')
cols[0].pyplot(fig1)
cols[1].write('Summary')
cols[1].write(('Min =', conv_len.num_words.min()))
cols[1].write(('25th percentile =', np.round(np.percentile(conv_len.num_words, 25), 1)))
cols[1].write(('Avg =', np.round(conv_len.num_words.mean(), 1)))
cols[1].write(('Median =', conv_len.num_words.median()))
cols[1].write(('75th percentile =', np.round(np.percentile(conv_len.num_words, 75), 1)))
cols[1].write(('Max =', conv_len.num_words.max()))
cols[1].write('')
cols[1].write('Conversations with less than 100 words:')
cols[1].write(conv_len[conv_len.num_words < 100].sort_values('num_words'))

# ---------
# st.subheader('Most frequent words')
#
# st.image(wc.to_array())
#
# cols = st.columns(3)
# cols[0].write(words)
# cols[1].dataframe(pd.DataFrame({'perc of words': range(10, 110, 10), 'num of freq words': num_mfw}))
# cols[2].dataframe(pd.DataFrame({'num of top words': np.array([10, 50, 100, 150, 200, 300, 400, 500, 600, 1000]),
#                                 'perc of words': perc_mfw}))

# ----------------------------------------------------------------------------------------------------------------------

st.header('Notes on fusion')
st.write('The following ways are considered for combining the mfw method with the voc output:')
st.write('1. assume that mfw and voc LR are independent and multiply them (multi)')
st.write('2a. apply classifier using as input the mfc and the voc score, then calibrate the resulted score (combi_a)')
st.write('2b. apply classifier using as input: the mfc score, the voc score, and their product, then calibrate '
         'the resulted score (combi_b)')
st.write('3. use the voc score as additional feature to the mfw input vector (feat)')
st.write('4. per class, fit bivariate normal distribution on the mfw scorers and voc output (biva)')

# ----------------------------------------------------------------------------------------------------------------------
st.header('Parameter settings')
cols = st.columns(3)
cols[0].write('For VOCALISE:')
cols[0].text('recording device = ' + param['recording_device'] + '\n'
             'calibrator =  ' + param['voc_calibrator'] + '\n'
             'preprocessor used =  ' + param['voc_preprocessor'])

cols[1].write('For authorship:')
cols[1].text('num of frequent words = ' + str(param['num_mfw']) + '\n'
             'min num of words in a conversation = ' + str(param['min_num_words']) + '\n'
             'preprocessor used = ' + param['mfw_preprocessor'] + '\n'
             'distance = ' + param['mwf_distance'] + '\n'
             'classifier = ' + param['mwf_classifier'] + '\n'
             'calibrator = ' + param['mwf_calibrator'])

cols[2].write('General:')
cols[2].text('max num of pairs draw per class (train) = ' + str(param['max_pairs_per_class_train']) + '\n'
             'max num of pairs draw per class (test) = ' + str(param['max_pairs_per_class_test']) + '\n'
             'perc of speakers used for test set (from the 223) = ' + str(param['test_speakers_perc']) + '% \n'
             'repeats = ' + str(param['repeats']) + '\n'
             'fusion classifier = ' + param['fusion_classifier'] + '\n'
             'fusion calibrator = ' + param['fusion_calibrator'] + '')

st.write('We note that there are sessions with no telephone recordings. The transcriptions of those sessions were '
         'excluded from the analysis.')

# ----------------------------------------------------------------------------------------------------------------------

st.header('Number of pairs used for training and testing')
st.write(num_pairs)

# ----------------------------------------------------------------------------------------------------------------------

st.header('Results')

# ---------
st.subheader('Performance metrics')
st.write(res_df)
st.write(' ')
st.write(' ')
# st.pyplot(sns.catplot(x="method", y="value", col='metric', data=metrics_per_repeat, sharey=False))
st.pyplot(sns.catplot(x="method", y="value", col='metric', data=metrics_per_repeat, kind="box"))

# ---------
st.subheader('Checking the log10LRs')

# st.write('Percentiles per method (all runs)')
# perc_df = pd.DataFrame([percentiles_for_df('mfw', predictions.lrs_mfw),
#                         percentiles_for_df('voc', predictions.lrs_voc),
#                         percentiles_for_df('multi', predictions.lrs_multi),
#                         percentiles_for_df('combi_a', predictions.lrs_comb_a),
#                         percentiles_for_df('combi_b', predictions.lrs_comb_b),
#                         percentiles_for_df('feat', predictions.lrs_feat),
#                         percentiles_for_df('biva', predictions.lrs_biva)],
#                        columns=['method', 'min', '25th', '50th', '75th', 'max'])
# st.write(perc_df)
st.write('')
st.write('')
st.write('Histograms, pav plots and tippett plots (all runs)')
cols = st.columns(4)
for key, m in {0: 'lrs_mfw', 1: 'lrs_voc', 2: 'lrs_multi', 3: 'lrs_comb_a', 4: 'lrs_comb_b',
               5: 'lrs_feat', 6: 'lrs_biva', 7: 'empty'}.items():
    if key == 7:
        key = key % 4
        empty_plot = plt.imread(f'{main_path}/empty.png')
        cols[key].image(empty_plot)
    else:
        key = key % 4
        cols[key].pyplot(lr_histogram(predictions[m], predictions['y'], title=m.replace('lrs_', '')))

for key, m in {0: 'lrs_mfw', 1: 'lrs_voc', 2: 'lrs_multi', 3: 'lrs_comb_a', 4: 'lrs_comb_b',
               5: 'lrs_feat', 6: 'lrs_biva', 7: 'empty'}.items():
    if key == 7:
        key = key % 4
        empty_plot = plt.imread(f'{main_path}/empty.png')
        cols[key].image(empty_plot)
    else:
        key = key % 4
        pav_plot = plt.imread(f'{main_path}/pav_{m}.png')
        cols[key].image(pav_plot)

for key, m in {0: 'lrs_mfw', 1: 'lrs_voc', 2: 'lrs_multi', 3: 'lrs_comb_a', 4: 'lrs_comb_b',
               5: 'lrs_feat', 6: 'lrs_biva', 7: 'empty'}.items():
    if key == 7:
        key = key % 4
        empty_plot = plt.imread(f'{main_path}/empty.png')
        cols[key].image(empty_plot)
    else:
        key = key % 4
        tippett_plot = plt.imread(f'{main_path}/tippett_{m}.png')
        cols[key].image(tippett_plot)

# ---------
st.subheader('Error analysis')

cols = st.columns(4)
option1 = cols[0].selectbox('Method A', ('mfw', 'voc', 'multi', 'comb_a', 'comb_b', 'feat', 'biva'), index=1)
option2 = cols[1].selectbox('Method B', ('mfw', 'voc', 'multi', 'comb_a', 'comb_b', 'feat', 'biva'), index=3)
# option3 = cols[2].selectbox('Class', ('any', 'same speaker', 'different speakers'))
# option4 = cols[3].selectbox('Predictions', ('all', 'incorrect by A only', 'incorrect by B only',
#                                             'incorrect by both methods'))

pred_subset = predictions

st.write(' ')
st.write(' ')
st.write('Confusion matrices (true labels -> rows, predicted labels -> columns):')
cols = st.columns(2)

cols[0].write('for ' + option1)
cols[0].write(confusion_matrix(pred_subset['y'].astype(str),
                               pd.Series(np.where(pred_subset['lrs_' + option1] > 1, 1, 0)).astype(str)))
st.write('')
st.write('')
grouped_loc = metrics_per_category(['spk_locations'], 'lrs_' + option1, 'y', pred_subset)
grouped_env = metrics_per_category(['spk_environments'], 'lrs_' + option1, 'y', pred_subset)
grouped_loc_env = metrics_per_category(['spk_locations', 'spk_environments'], 'lrs_' + option1,
                                       'y', pred_subset)
grouped_words = metrics_per_category(['spk_num_words'], 'lrs_' + option1, 'y', pred_subset)
grouped_2_lan = metrics_per_category(['second_language_A', 'second_language_B'], 'lrs_' + option1, 'y', pred_subset)

if option1 != option2:
    pred_subset['pred_B'] = pd.Series(np.where(pred_subset['lrs_' + option2] > 1, 1, 0))
    cols[1].write('for ' + option2)
    cols[1].write(confusion_matrix(pred_subset['y'].astype(str), pred_subset['pred_B'].astype(str)))

    temp_loc = metrics_per_category(['spk_locations'], 'lrs_' + option2, 'y', pred_subset)
    grouped_loc.rename(columns={'cllr': 'cllr' + '_' + option1, 'eer': 'eer' + '_' + option1}, inplace=True)
    temp_loc.rename(columns={'cllr': 'cllr' + '_' + option2, 'eer': 'eer' + '_' + option2}, inplace=True)
    grouped_loc = grouped_loc.merge(temp_loc, on=['spk_locations', 'total_ds', 'total_ss'])

    temp_env = metrics_per_category(['spk_environments'], 'lrs_' + option2, 'y', pred_subset)
    grouped_env.rename(columns={'cllr': 'cllr' + '_' + option1, 'eer': 'eer' + '_' + option1}, inplace=True)
    temp_env.rename(columns={'cllr': 'cllr' + '_' + option2, 'eer': 'eer' + '_' + option2}, inplace=True)
    grouped_env = grouped_env.merge(temp_env, on=['spk_environments', 'total_ds', 'total_ss'])

    temp_loc_env = metrics_per_category(['spk_locations', 'spk_environments'],
                                        'lrs_' + option2, 'y', pred_subset)
    grouped_loc_env.rename(columns={'cllr': 'cllr' + '_' + option1, 'eer': 'eer' + '_' + option1}, inplace=True)
    temp_loc_env.rename(columns={'cllr': 'cllr' + '_' + option2, 'eer': 'eer' + '_' + option2}, inplace=True)
    grouped_loc_env = grouped_loc_env.merge(temp_loc_env,
                                            on=['spk_locations', 'spk_environments', 'total_ds', 'total_ss'])

    temp_words = metrics_per_category(['spk_num_words'], 'lrs_' + option2, 'y', pred_subset)
    grouped_words.rename(columns={'cllr': 'cllr' + '_' + option1, 'eer': 'eer' + '_' + option1}, inplace=True)
    temp_words.rename(columns={'cllr': 'cllr' + '_' + option2, 'eer': 'eer' + '_' + option2}, inplace=True)
    grouped_words = grouped_words.merge(temp_words, on=['spk_num_words', 'total_ds', 'total_ss'])

    temp_2_lan = metrics_per_category(['second_language_A', 'second_language_B'], 'lrs_' + option2, 'y', pred_subset)
    grouped_2_lan.rename(columns={'cllr': 'cllr' + '_' + option1, 'eer': 'eer' + '_' + option1}, inplace=True)
    temp_2_lan.rename(columns={'cllr': 'cllr' + '_' + option2, 'eer': 'eer' + '_' + option2}, inplace=True)
    grouped_2_lan = grouped_2_lan.merge(temp_2_lan,
                                        on=['second_language_A', 'second_language_B', 'total_ds', 'total_ss'])

st.write('Metrics depending location')
st.write(grouped_loc)
st.write(' ')
st.write(' ')
st.write('Metrics depending environment')
st.write(grouped_env)
st.write(' ')
st.write(' ')
st.write('Metrics depending location and environment')
st.write(grouped_loc_env)
st.write(' ')
st.write(' ')
st.write('Metrics depending num of words')
st.write(grouped_words)
st.write(' ')
st.write(' ')
st.write('Metrics depending second language')
st.write(grouped_2_lan)


tt = metrics_for_plotting(['spk_num_words'], pred_subset)
st.pyplot(sns.catplot(x="value", y="spk_num_words",hue="method", col="metric",data=tt, kind="bar", orient="h"))
print('hello')

# if option3 == 'same speaker':
#     pred_to_show = pred_subset[pred_subset['y'] == 1]
# elif option3 == 'different speakers':
#     pred_to_show = pred_subset[pred_subset['y'] == 0]
# else:
#     pred_to_show = pred_subset
#
# if option4 == 'incorrect by A only':
#     pred_to_show = pred_to_show[(((pred_to_show['lrs_' + option1] > 1) & (pred_to_show['y'] != 1)) |
#                                  ((pred_to_show['lrs_' + option1] <= 1) & (pred_to_show['y'] != 0))) &
#                                 (((pred_to_show['lrs_' + option2] > 1) & (pred_to_show['y'] == 1)) |
#                                  ((pred_to_show['lrs_' + option2] <= 1) & (pred_to_show['y'] == 0)))]
# elif option4 == 'incorrect by B only':
#     pred_to_show = pred_to_show[(((pred_to_show['lrs_' + option1] > 1) & (pred_to_show['y'] == 1)) |
#                                  ((pred_to_show['lrs_' + option1] <= 1) & (pred_to_show['y'] == 0))) &
#                                 (((pred_to_show['lrs_' + option2] > 1) & (pred_to_show['y'] != 1)) |
#                                  ((pred_to_show['lrs_' + option2] <= 1) & (pred_to_show['y'] != 0)))]
# elif option4 == 'incorrect by both methods':
#     pred_to_show = pred_to_show[(((pred_to_show['lrs_' + option1] > 1) & (pred_to_show['y'] != 1)) |
#                                  ((pred_to_show['lrs_' + option1] <= 1) & (pred_to_show['y'] != 0))) &
#                                 (((pred_to_show['lrs_' + option2] > 1) & (pred_to_show['y'] != 1)) |
#                                  ((pred_to_show['lrs_' + option2] <= 1) & (pred_to_show['y'] != 0)))]
#
# st.write(' ')
# st.write(' ')
# st.write('num of rows = ', pred_to_show.shape[0])
# st.write(pred_to_show)
