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


# percentiles
def percentiles_for_df(name, arr):
    log_arr = np.log10(arr)
    temp = np.percentile(log_arr, range(0, 101, 25))
    temp = [round(i, 4) if i < 1 else round(i, 1) for i in temp]
    return [name] + temp


def lr_histogram(lrs, y, bins=20, title=''):
    """
    plots the 10log lrs
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
    conv_metadata = pd.DataFrame({'session': [2, 4, 6, 8], 'location': ['indoor', 'indoor', 'outdoor', 'indoor'],
                                  'environment': ['quiet', 'noisy', 'quiet', 'noisy']})

    # performance results (high level)
    res_df = pd.DataFrame([['mfw', 0.516, 0.85, 0.154, 0.839, 0.587], ['voc', 0.139, 0.961, 0.037, 0.966, 0.858],
                           ['multi', 0.1, 0.974, 0.027, 0.971, 0.907], ['combi', 0.116, 0.971, 0.032, 0.965, 0.897],
                           ['featu', 0.115, 0.971, 0.028, 0.972, 0.894]],
                          columns=['method', 'cllr', 'accuracy', 'eer', 'recall', 'precision'])

    # predictions
    with open(f'{main_path}/predictions_per_repeat_prep_gauss_2500.json') as f:
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

    # performance results per repeat
    metric_functions = {'cllr': cllr, 'acc': accuracy, 'eer': eer}
    metrics_per_repeat = pd.DataFrame()

    for col in ['lrs_mfw', 'lrs_voc', 'lrs_multi', 'lrs_comb', 'lrs_feat']:

        # generate pav plots
        lir.plot_pav(predictions[col], predictions['y'], savefig=f'{main_path}/pav_{col}.png')

        # calculate cllr, accuracy and eer per repeat
        for m, f in metric_functions.items():
            temp = predictions.groupby('repeat')[['y', col]].apply(lambda x: f(x[col], x['y'])).to_frame()
            temp.reset_index(level=0, inplace=True)
            temp['metric'] = m
            temp['method'] = col.replace('lrs_', '')
            temp.rename(columns={0: 'value'}, inplace=True)
            temp = temp[['method', 'repeat', 'metric', 'value']]
            metrics_per_repeat = metrics_per_repeat.append(temp, ignore_index=True)

    # add info on predictions
    predictions['location_A'] = np.where(predictions['conv_A'].str.endswith(('2', '4')), 'indoor', 'outdoor')
    predictions['environment_A'] = np.where(predictions['conv_A'].str.endswith(('2', '6')), 'quiet', 'noisy')
    predictions['location_A'] = predictions['location_A'] + '_' + predictions['environment_A']
    predictions['location_B'] = np.where(predictions['conv_B'].str.endswith(('2', '4')), 'indoor', 'outdoor')
    predictions['environment_B'] = np.where(predictions['conv_B'].str.endswith(('2', '6')), 'quiet', 'noisy')
    predictions['location_B'] = predictions['location_B'] + '_' + predictions['environment_B']
    predictions = predictions.merge(conv_len.add_suffix('_A'), left_on='conv_A', right_on='conv_id_A')
    predictions = predictions.merge(conv_len.add_suffix('_B'), left_on='conv_B', right_on='conv_id_B')
    predictions = predictions.merge(spk_metadata.add_suffix('_A'), left_on='spk_A', right_on='spk_id_A')
    predictions = predictions.merge(spk_metadata.add_suffix('_B'), left_on='spk_B', right_on='spk_id_B')
    predictions.drop(['environment_A', 'environment_B', 'spk_id_A', 'spk_id_B', 'conv_id_A', 'conv_id_B'], axis=1,
                     inplace=True)



# ----------------------------------------------------------------------------------------------------------------------
st.set_page_config(layout="wide")

st.title('Fusion of a speaker and a author verification system - Exploring the predictions')
st.markdown("""---""")

# ----------------------------------------------------------------------------------------------------------------------
st.header('FRIDA dataset')
st.write('Based on the transcriptions, we are working with ', len(spks), ' speakers. In the metadata file, there are ',
         len(spks_in_meta), ' speakers with ', len(spks_in_meta_with_no_trans), ' of them having no transcriptions',
         'Additionally, we have ', len(spks_not_in_meta), ' speakers with transcriptions with no record in the'
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
cols[0].text('recording device = telephone \n'
             'calibrator = Logit \n'
             'preprocessor used = MinMaxScaler')

cols[1].write('For authorship:')
cols[1].text('num of frequent words = 200 \n'
             'min num of words in a conversation = 1 \n'
             'preprocessor used = percentile rank \n'
             'distance = Bray-Curtis \n'
             'classifier = Logistic Regression \n'
             'calibrator = KDE')

cols[2].write('General:')
cols[2].text('max num of pairs draw per class = 2500 (for both train and test set) \n'
             'perc of speakers used for test set = 10% (from the 223) \n'
             'repeats = 10 \n'
             'fusion classifier = Logistic Regression \n'
             'fusion calibrator = KDE')

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
option3 = cols[3].selectbox('Class', ('any', 'same speaker', 'different speakers'))
option4 = cols[4].selectbox('Predictions', ('all', 'incorrect by A only', 'incorrect by B only',
                                            'incorrect by both methods'))

if option0 != 'all':
    pred_to_show = predictions[predictions['repeat'] == option0]
else:
    pred_to_show = predictions

st.write(' ')
st.write(' ')
st.write('Confusion matrix (true labels -> rows, predicted labels -> columns):')
cols = st.columns(2)
pred_A = pd.Series(np.where(pred_to_show['lrs_' + option1] > 1, 1, 0)).astype(str)
cols[0].write('for '+option1)
cols[0].write(confusion_matrix(pred_to_show['y'].astype(str), pred_A))
if option1 != option2:
    pred_B = pd.Series(np.where(pred_to_show['lrs_' + option2] > 1, 1, 0)).astype(str)
    cols[1].write('for '+option2)
    cols[1].write(confusion_matrix(pred_to_show['y'].astype(str), pred_B))


if option3 == 'same speaker':
    pred_to_show = pred_to_show[pred_to_show['y'] == 1]
elif option3 == 'different speakers':
    pred_to_show = pred_to_show[pred_to_show['y'] == 0]


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
