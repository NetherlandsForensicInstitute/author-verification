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
runs = ['prep_gauss_2500']
specific_run = runs[0]
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
    return fpr[np.nanargmin(np.absolute((fnr - fpr)))]


# accuracy
def accuracy(lrs, y):
    return np.mean((lrs > 1) == y)


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


def accuracies_per_category(cols_to_group, lrs_col, y, df):
    """
    :param cols_to_group = list of column names
    :param lrs_col = column name of the lrs
    :param y = column name of the ground truth (binary 0/1)
    :param df = dataframe with the mentioned columns

    it returns a df where each comb of values in cols_to_group the counts and the accuracies for class 0 and 1
    """
    # counts
    grouped_df = df.groupby(cols_to_group)[y].agg([lambda x: sum(x == 0), lambda x: sum(x)]).reset_index()
    grouped_df.rename(columns={'<lambda_0>': "total_0", '<lambda_1>': "total_1"}, inplace=True)

    # accuracies
    df['pred'] = pd.Series(np.where((df[lrs_col] > 1) & (df[y] == 1), 1, np.where((df[lrs_col] <= 1) &
                                                                                  (df[y] == 0), 0, 2)))
    grouped_pred = df.groupby(cols_to_group)['pred'].agg([lambda x: sum(x < 2), lambda x: sum(x == 0),
                                                          lambda x: sum(x == 1)]).reset_index()
    new_col_name = lrs_col.replace('lrs_', '') + '_acc'
    new_col_name_0 = lrs_col.replace('lrs_', '') + '_acc_0'
    new_col_name_1 = lrs_col.replace('lrs_', '') + '_acc_1'
    grouped_pred.rename(columns={'<lambda_0>': new_col_name, '<lambda_1>': new_col_name_0,
                                 '<lambda_2>': new_col_name_1}, inplace=True)

    grouped_df = grouped_df.merge(grouped_pred, on=cols_to_group)
    grouped_df[new_col_name] = round(grouped_df[new_col_name] / (grouped_df['total_0'] + grouped_df['total_1']), 4)
    grouped_df[new_col_name_0] = round(grouped_df[new_col_name_0] / grouped_df['total_0'], 4)
    grouped_df[new_col_name_1] = round(grouped_df[new_col_name_1] / grouped_df['total_1'], 4)

    return grouped_df.sort_values([new_col_name, new_col_name_1, new_col_name_0])


if __name__ == '__main__':

    # load and prep data

    # words and their frequencies (based on whole dataset)
    words = pd.read_json(f'{main_path}/word_frequencies.json', orient="index")
    words.rename(columns={0: "freq"}, inplace=True)
    words.sort_values(by='freq', ascending=False, inplace=True)
    top_200 = words.index[:200].tolist()

    wc = WordCloud(width=2500, height=400)
    wc.generate_from_frequencies(frequencies=words.to_dict()['freq'])

    num_mfw, perc_mfw = most_frequent(words)

    # length of the conversations (excl. words that are non-existing, incomplete, distorted or unclear)
    conv_len = pd.read_json(f'{main_path}/conversation_length.json', orient="index")
    conv_len.reset_index(inplace=True)
    conv_len.rename(columns={'index': 'conv_id', 0: "num_words"}, inplace=True)
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

    spk_metadata['birth_place_father'] = spk_metadata['birth_place_father']. \
        str.replace('[mM]oroc[coan]*', 'Morocco'). \
        str.replace('Turk[eyish]*', 'Turkey'). \
        str.replace('Neatherlands', 'Netherlands', regex=False). \
        str.replace('Holland', 'Netherlands', regex=False). \
        str.replace('Dutch', 'Netherlands', regex=False). \
        str.replace('Amsterdam', 'Netherlands', regex=False)
    spk_metadata['birth_place_mother'] = spk_metadata['birth_place_mother']. \
        str.replace('[mM]oroc[coan]*', 'Morocco'). \
        str.replace('Tu[r]{0,1}k[eyish]*', 'Turkey'). \
        str.replace('Neatherlands', 'Netherlands', regex=False). \
        str.replace('Hol[la][la]n[d]{0,1}', 'Netherlands'). \
        str.replace('Dutch', 'Netherlands', regex=False)
    spk_metadata['second_language'] = spk_metadata['second_language']. \
        str.replace('[mM]o[rc]oc[coan]*', 'Moroccan'). \
        str.replace('Tu[r]{0,1}k[eyish]*', 'Turkish'). \
        str.replace('Enligsh', 'English', regex=False)

    # create conversation metadata
    conv_metadata = pd.DataFrame({'session': [2, 4, 6, 8], 'location': ['indoor', 'indoor', 'outdoor', 'outdoor'],
                                  'environment': ['quiet', 'noisy', 'quiet', 'noisy']})

    # parameters
    with open(f'{main_path}/{param_file}') as p:
        param = json.load(p)

    # predictions
    with open(f'{main_path}/{prediction_file}') as f:
        per_repeat = json.load(f)

    predictions = pd.DataFrame()
    num_pairs = pd.DataFrame()

    for key in per_repeat.keys():
        train_same, train_diff = sum(np.array(per_repeat[key]['train']) == 1), sum(
            np.array(per_repeat[key]['train']) == 0)
        test_same, test_diff = sum(np.array(per_repeat[key]['y']) == 1), sum(np.array(per_repeat[key]['y']) == 0)
        temp_num_df = pd.DataFrame({'repeat': key, 'train_same': train_same, 'train_diff': train_diff,
                                    'train_all': len(per_repeat[key]['train']), 'test_same': test_same,
                                    'test_diff': test_diff, 'test_all': len(per_repeat[key]['y'])}, index=[0])

        conv_A = [pair[0] for pair in per_repeat[key]['pairs']]
        spk_A = [conv_id[:-2] for conv_id in conv_A]
        conv_B = [pair[1] for pair in per_repeat[key]['pairs']]
        spk_B = [conv_id[:-2] for conv_id in conv_B]
        lrs_multi = [a * b for a, b in zip(per_repeat[key]['lrs_mfw'], per_repeat[key]['lrs_voc'])]

        temp_df = pd.DataFrame({'repeat': key, 'spk_A': spk_A, 'spk_B': spk_B, 'conv_A': conv_A, 'conv_B': conv_B,
                                'y': per_repeat[key]['y'], 'lrs_mfw': per_repeat[key]['lrs_mfw'],
                                'lrs_voc': per_repeat[key]['lrs_voc'], 'lrs_multi': lrs_multi,
                                'lrs_comb': per_repeat[key]['lrs_comb'], 'lrs_feat': per_repeat[key]['lrs_feat']})
        temp_df['y'] = temp_df['y'].astype('Int64')
        predictions = predictions.append(temp_df, ignore_index=True)
        num_pairs = num_pairs.append(temp_num_df, ignore_index=True)

    # performance results on a high level and per repeat
    res_df = pd.DataFrame(columns=['method', 'cllr', 'accuracy', 'eer', 'recall', 'precision'])
    metric_functions = {'cllr': cllr, 'acc': accuracy, 'eer': eer}
    metrics_per_repeat = pd.DataFrame()

    for col in ['lrs_mfw', 'lrs_voc', 'lrs_multi', 'lrs_comb', 'lrs_feat']:

        method = col.replace('lrs_', '')

        # generate pav plots
        lir.plot_pav(predictions[col], predictions['y'], savefig=f'{main_path}/pav_{col}.png')

        # cllr, accuracy, eer, recall and precision on a high level
        res_df_row = pd.Series([method, cllr(predictions[col], predictions['y']), accuracy(predictions[col], predictions['y']),
                      eer(predictions[col], predictions['y']), recall(predictions[col], predictions['y']),
                      precision(predictions[col], predictions['y'])], index=res_df.columns)
        res_df = res_df.append(res_df_row, ignore_index=True)

        # calculate cllr, accuracy and eer per repeat
        for m, f in metric_functions.items():

            temp = predictions.groupby('repeat')[['y', col]].apply(lambda x: f(x[col], x['y'])).to_frame()
            temp.reset_index(level=0, inplace=True)
            temp['metric'] = m
            temp['method'] = method
            temp.rename(columns={0: 'value'}, inplace=True)
            temp = temp[['method', 'repeat', 'metric', 'value']]
            metrics_per_repeat = metrics_per_repeat.append(temp, ignore_index=True)

    # add info on predictions
    predictions['location_A'] = np.where(predictions['conv_A'].str.endswith(('2', '4')), 'indoor', 'outdoor')
    predictions['environment_A'] = np.where(predictions['conv_A'].str.endswith(('2', '6')), 'quiet', 'noisy')
    predictions['location_B'] = np.where(predictions['conv_B'].str.endswith(('2', '4')), 'indoor', 'outdoor')
    predictions['environment_B'] = np.where(predictions['conv_B'].str.endswith(('2', '6')), 'quiet', 'noisy')
    predictions = predictions.merge(conv_len.add_suffix('_A'), left_on='conv_A', right_on='conv_id_A')
    predictions = predictions.merge(conv_len.add_suffix('_B'), left_on='conv_B', right_on='conv_id_B')
    predictions = predictions.merge(spk_metadata.add_suffix('_A'), left_on='spk_A', right_on='spk_id_A')
    predictions = predictions.merge(spk_metadata.add_suffix('_B'), left_on='spk_B', right_on='spk_id_B')
    predictions.drop(['spk_id_A', 'spk_id_B', 'conv_id_A', 'conv_id_B'], axis=1, inplace=True)

# ----------------------------------------------------------------------------------------------------------------------
st.set_page_config(layout="wide")

st.title('Fusion of a speaker and a author verification system - Exploring the predictions')
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
cols[1].write(spk_metadata.birth_place_father.value_counts(ascending=False))
cols[2].write(spk_metadata.birth_place_mother.value_counts(ascending=False))
cols[3].write(spk_metadata.second_language.value_counts(ascending=False))

# ---------
st.subheader('Sessions')
st.write('We are focusing on the sessions 2, 4, 6, and 8, as those are the sessions for which a transcription is '
         'available. The table belows shows were the conversation took place and under which background conditions.')
st.write(conv_metadata)

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
st.subheader('Most frequent words')

st.image(wc.to_array())

cols = st.columns(3)
cols[0].write(words)
cols[1].dataframe(pd.DataFrame({'perc of words': range(10, 110, 10), 'num of freq words': num_mfw}))
cols[2].dataframe(pd.DataFrame({'num of top words': np.array([10, 50, 100, 150, 200, 300, 400, 500, 600, 1000]),
                                'perc of words': perc_mfw}))

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
st.text('(weighted average over all runs)')
st.write(' ')
st.write(' ')
st.pyplot(sns.catplot(x="method", y="value", col='metric', data=metrics_per_repeat, sharey=False))

# ---------
st.subheader('Checking the log10LRs')

st.write('Percentiles per method (all runs)')
perc_df = pd.DataFrame([percentiles_for_df('mfw', predictions.lrs_mfw),
                        percentiles_for_df('voc', predictions.lrs_voc),
                        percentiles_for_df('multi', predictions.lrs_multi),
                        percentiles_for_df('combi', predictions.lrs_comb),
                        percentiles_for_df('feat', predictions.lrs_feat)],
                       columns=['method', 'min', '25th', '50th', '75th', 'max'])
st.write(perc_df)
st.write('')
st.write('')
st.write('Histograms and pav plots (all runs)')
cols = st.columns(5)
for key, m in {0: 'lrs_mfw', 1: 'lrs_voc', 2: 'lrs_multi', 3: 'lrs_comb', 4: 'lrs_feat'}.items():
    cols[key].pyplot(lr_histogram(predictions[m], predictions['y'], title=m.replace('lrs_', '')))
    pav_plot = plt.imread(f'{main_path}/pav_{m}.png')
    cols[key].image(pav_plot)

# ---------
st.subheader('Error analysis')

cols = st.columns(5)
option0 = cols[0].selectbox('Repeat', ('all', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
option1 = cols[1].selectbox('Method A', ('mfw', 'voc', 'multi', 'comb', 'feat'))
option2 = cols[2].selectbox('Method B', ('mfw', 'voc', 'multi', 'comb', 'feat'))
option3 = cols[3].selectbox('Class (affects only the dataframe)', ('any', 'same speaker', 'different speakers'))
option4 = cols[4].selectbox('Predictions (affects only the dataframe)', ('all', 'incorrect by A only',
                                                                         'incorrect by B only',
                                                                         'incorrect by both methods'))

if option0 != 'all':
    pred_subset = predictions[predictions['repeat'] == option0]
else:
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
grouped_loc = accuracies_per_category(['location_A', 'location_B'], 'lrs_' + option1, 'y', pred_subset)
grouped_env = accuracies_per_category(['environment_A', 'environment_B'], 'lrs_' + option1, 'y', pred_subset)
grouped_2_lan = accuracies_per_category(['second_language_A', 'second_language_B'], 'lrs_' + option1, 'y', pred_subset)

if option1 != option2:
    pred_subset['pred_B'] = pd.Series(np.where(pred_subset['lrs_' + option2] > 1, 1, 0))
    cols[1].write('for '+option2)
    cols[1].write(confusion_matrix(pred_subset['y'].astype(str), pred_subset['pred_B'].astype(str)))

    temp_loc = accuracies_per_category(['location_A', 'location_B'], 'lrs_' + option2, 'y', pred_subset)
    grouped_loc = grouped_loc.merge(temp_loc, on=['location_A', 'location_B', 'total_0', 'total_1'])

    temp_env = accuracies_per_category(['environment_A', 'environment_B'], 'lrs_' + option2, 'y', pred_subset)
    grouped_env = grouped_env.merge(temp_env, on=['environment_A', 'environment_B', 'total_0', 'total_1'])

    temp_2_lan = accuracies_per_category(['second_language_A', 'second_language_B'], 'lrs_' + option2, 'y', pred_subset)
    grouped_2_lan = grouped_2_lan.merge(temp_2_lan, on=['second_language_A', 'second_language_B', 'total_0', 'total_1'])

st.write('accuracies depending location')
st.write(grouped_loc)
st.write(' ')
st.write(' ')
st.write('accuracies depending environment')
st.write(grouped_env)
st.write(' ')
st.write(' ')
st.write('accuracies depending second language')
st.write(grouped_2_lan)

if option3 == 'same speaker':
    pred_to_show = pred_subset[pred_subset['y'] == 1]
elif option3 == 'different speakers':
    pred_to_show = pred_subset[pred_subset['y'] == 0]
else:
    pred_to_show = pred_subset

if option4 == 'incorrect by A only':
    pred_to_show = pred_to_show[(((pred_to_show['lrs_' + option1] > 1) & (pred_to_show['y'] != 1)) |
                                 ((pred_to_show['lrs_' + option1] <= 1) & (pred_to_show['y'] != 0))) &
                                (((pred_to_show['lrs_' + option2] > 1) & (pred_to_show['y'] == 1)) |
                                 ((pred_to_show['lrs_' + option2] <= 1) & (pred_to_show['y'] == 0)))]
elif option4 == 'incorrect by B only':
    pred_to_show = pred_to_show[(((pred_to_show['lrs_' + option1] > 1) & (pred_to_show['y'] == 1)) |
                                 ((pred_to_show['lrs_' + option1] <= 1) & (pred_to_show['y'] == 0))) &
                                (((pred_to_show['lrs_' + option2] > 1) & (pred_to_show['y'] != 1)) |
                                 ((pred_to_show['lrs_' + option2] <= 1) & (pred_to_show['y'] != 0)))]
elif option4 == 'incorrect by both methods':
    pred_to_show = pred_to_show[(((pred_to_show['lrs_' + option1] > 1) & (pred_to_show['y'] != 1)) |
                                 ((pred_to_show['lrs_' + option1] <= 1) & (pred_to_show['y'] != 0))) &
                                (((pred_to_show['lrs_' + option2] > 1) & (pred_to_show['y'] != 1)) |
                                 ((pred_to_show['lrs_' + option2] <= 1) & (pred_to_show['y'] != 0)))]

st.write(' ')
st.write(' ')
st.write('num of rows = ', pred_to_show.shape[0])
st.write(pred_to_show)
