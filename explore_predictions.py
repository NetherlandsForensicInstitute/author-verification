import lir
import json
import os

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
import seaborn as sns

from sklearn.metrics import roc_curve, confusion_matrix

main_path = 'fisher/predictions'
info_path = 'fisher/info.txt'

exp = 1
runs = ['exp0', 'exp1', 'exp2', 'exp3']
filenames = ['_all_01fortest', '_all_stratified_01fortest', '_all_stratified_02fortest',
             '_all_stratified_02fortest5000']
specific_run = runs[exp]
update_files = False

# equal error rate
def eer(lrs, y):
    if len(y.value_counts()) > 1:
        fpr, tpr, threshold = roc_curve(list(y), list(lrs), pos_label=1)
        fnr = 1 - tpr
        try:
            return fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        except ValueError:
            return np.nan
    else:
        return np.inf #-np.mean(lrs[y == 0] > 1)  # False positive rate


# cllr
def cllr(lrs, y):
    if len(y.value_counts()) > 1:
        return lir.metrics.cllr(lrs, y)
    else:
        return np.inf


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


def metrics_per_category(cols_to_group, df):
    """
    parameters
    :param cols_to_group = list of column names
    :param df = dataframe with the mentioned columns including ground truth column named 'y'

    it returns two df: in long and wide format. wide format incl counts for all value combination in cols_to_group.
    both formats have the cllr and eer for all value combination in cols_to_group.
    """
    # counts
    cols = ['repeat'] + cols_to_group

    long_df = pd.DataFrame()
    for col in ['lrs_mfw', 'lrs_sv', 'lrs_multi', 'lrs_comb_a', 'lrs_comb_b', 'lrs_feat', 'lrs_biva']:
        cllr_df = df.groupby(cols)[['y', col]].apply(lambda x: cllr(x[col], x['y'])).to_frame().reset_index()
        cllr_df.rename(columns={0: "value"}, inplace=True)
        cllr_df = cllr_df.groupby(cols_to_group).agg('mean').reset_index()
        cllr_df['metric'] = 'cllr'
        cllr_df['method'] = col.replace('lrs_', '')
        cllr_df = cllr_df[['method'] + cols_to_group + ['metric', 'value']]
        long_df = long_df.append(cllr_df, ignore_index=True)

        eer_df = df.groupby(cols)[['y', col]].apply(lambda x: eer(x[col], x['y'])).to_frame().reset_index()
        eer_df.rename(columns={0: "value"}, inplace=True)
        eer_df = eer_df.groupby(cols_to_group).agg('mean').reset_index()
        eer_df['metric'] = 'eer'
        eer_df['method'] = col.replace('lrs_', '')
        eer_df = eer_df[['method'] + cols_to_group + ['metric', 'value']]
        long_df = long_df.append(eer_df, ignore_index=True)

    counts = df.groupby(cols)['y'].agg([lambda x: sum(x == 0), lambda x: sum(x)]).reset_index()
    counts.rename(columns={'<lambda_0>': "total_ds", '<lambda_1>': "total_ss"}, inplace=True)
    counts = counts.groupby(cols_to_group).agg('mean').reset_index()
    if 'repeat' in counts.columns:
        counts.drop(['repeat'], axis=1, inplace=True)

    wide_df = pd.pivot(long_df, index=cols_to_group, columns=['method', 'metric'], values='value').reset_index()
    wide_df.columns = ['{}/{}'.format(x[0], str(x[1])) if x[1] != '' else x[0] for x in wide_df.columns]
    wide_df = counts.merge(wide_df, on=cols_to_group)

    wide_df = wide_df[wide_df.columns[0:len(cols_to_group)+2].\
    append(wide_df.columns[wide_df.columns.str.contains('cllr')]).\
    append(wide_df.columns[wide_df.columns.str.contains('eer')])]

    wide_df.drop(wide_df[wide_df.total_ss < 1].index, inplace=True)

    return long_df, wide_df


if __name__ == '__main__':

    # load and prep data
    # length of the conversations (excl. words that are non-existing, incomplete, distorted or unclear)
    conv_len = pd.read_json(f'{main_path}/conversation_length.json', orient="index")
    conv_len.reset_index(inplace=True)
    conv_len.rename(columns={'index': 'conv_id', 0: "num_words"}, inplace=True)
    bins_words = [0, 622, 787, 934, 1117, 2350]
    names_words = [str(bins_words[i]) + '<n<=' + str(bins_words[i+1]) for i in range(len(bins_words)-1)]
    conv_len['num_words_range'] = pd.cut(conv_len['num_words'], bins_words, labels=names_words).astype('O')
    conv_len['conv_id'] = conv_len['conv_id'].str.split('_').str[1]
    
    to_ids = {}
    with open(info_path) as f:
        next(f)
        for line in f:
            filename, spk_id, trans, sig_grade, phset, phtyp, sx, dl, count = line.split('\t', 8)
            filename = filename.replace('fe_03_', '').replace('_a', 'a').replace('_b', 'b')
            to_ids[filename] = [spk_id, sig_grade, phset, phtyp, trans, sx, dl]

    metadata = pd.DataFrame.from_dict(to_ids, orient='index',
                                      columns=["spk_id", "sig_grade", "phset", "phtyp", "trans", "sx", "dl"])
    metadata.reset_index(inplace=True)


    # predictions
    pred_path = f'{main_path}/{specific_run}/predictions_clean.csv'
    num_pairs_path = f'{main_path}/{specific_run}/num_pairs.csv'
    if os.path.exists(pred_path) and update_files==False:
        predictions = pd.read_csv(pred_path)
        num_pairs = pd.read_csv(num_pairs_path)
    else:
        with open(f'{main_path}/{specific_run}/predictions_per_repeat{filenames[exp]}.json') as f:
            per_repeat = json.load(f)

        predictions = pd.DataFrame()
        num_pairs = pd.DataFrame()

        for key in per_repeat.keys():

            test_same, test_diff = sum(np.array(per_repeat[key]['y']) == 1), sum(np.array(per_repeat[key]['y']) == 0)
            temp_num_df = pd.DataFrame({'repeat': key, 'test_same': test_same, 'test_diff': test_diff,
                                        'test_all': len(per_repeat[key]['y'])}, index=[0])

            conv_A = [pair.split('|')[0] for pair in per_repeat[key]['pairs']]
            conv_B = [pair.split('|')[1] for pair in per_repeat[key]['pairs']]
            lrs_multi = [a * b for a, b in zip(per_repeat[key]['lrs_mfw'], per_repeat[key]['lrs_sv'])]

            temp_df = pd.DataFrame({'repeat': key, 'conv_A': conv_A, 'conv_B': conv_B,
                                    'y': per_repeat[key]['y'], 'lrs_mfw': per_repeat[key]['lrs_mfw'],
                                    'lrs_sv': per_repeat[key]['lrs_sv'], 'lrs_multi': lrs_multi,
                                    'lrs_comb_a': per_repeat[key]['lrs_comb_a'],
                                    'lrs_comb_b': per_repeat[key]['lrs_comb_b'],
                                    'lrs_feat': per_repeat[key]['lrs_feat'], 'lrs_biva': per_repeat[key]['lrs_biva']})
            temp_df['y'] = temp_df['y'].astype('Int64')
            predictions = predictions.append(temp_df, ignore_index=True)
            num_pairs = num_pairs.append(temp_num_df, ignore_index=True)

        # add info on predictions
        predictions = predictions.merge(conv_len.add_suffix('_A'), left_on='conv_A', right_on='conv_id_A')
        predictions = predictions.merge(conv_len.add_suffix('_B'), left_on='conv_B', right_on='conv_id_B')
        predictions = predictions.merge(metadata.add_suffix('_A'), left_on='conv_A', right_on='index_A')
        predictions = predictions.merge(metadata.add_suffix('_B'), left_on='conv_B', right_on='index_B')
        predictions.drop(['conv_id_A', 'conv_id_B', 'index_A', 'index_B'], axis=1, inplace=True)

        predictions['spk_sx'] = np.where(predictions['sx_A'] == predictions['sx_B'],
                                                predictions['sx_A'], 'm vs f')
        predictions['spk_dl'] = np.where(predictions['dl_A'] == predictions['dl_B'],
                                                   predictions['dl_A'], 'a vs o')

        predictions['spk_num_words'] = predictions[['num_words_range_A', 'num_words_range_B']].values.tolist()
        predictions['spk_num_words'] = predictions['spk_num_words'].apply(sorted)
        predictions['spk_num_words'] = predictions['spk_num_words']. \
            apply(lambda x: x[0] if x[0] == x[1] else x[0] + ' vs ' + x[1])

        predictions['words_diff'] = abs(predictions['num_words_A'] - predictions['num_words_B'])
        bins_diff = np.percentile(predictions.words_diff, range(0, 101, 20))
        names_diff = [str(i) + '.' + str(int(bins_diff[i])) + '-' + str(int(bins_diff[i + 1])) for i in
                      range(len(bins_diff) - 1)]
        predictions['words_diff_range'] = pd.cut(predictions['words_diff'], bins_diff, labels=names_diff).astype('O')

        predictions.to_csv(pred_path, index=False)
        num_pairs.to_csv(num_pairs_path, index=False)

    # performance results on a high level and per repeat
    metrics_path = f'{main_path}/{specific_run}/metrics_per_repeat.csv'
    if os.path.exists(metrics_path):
        metrics_per_repeat = pd.read_csv(metrics_path)
    else:
        metric_functions = {'cllr': cllr, 'eer': eer}
        metrics_per_repeat = pd.DataFrame()

        for col in ['lrs_mfw', 'lrs_sv', 'lrs_multi', 'lrs_comb_a', 'lrs_comb_b', 'lrs_feat', 'lrs_biva']:

            method = col.replace('lrs_', '')

            # with lir.plotting.savefig(f'{main_path}/{specific_run}/pav_{col}.png') as ax:
            #     ax.pav(predictions[col].array, predictions['y'].array)
            #
            # with lir.plotting.savefig(f'{main_path}/{specific_run}/tippett_{col}.png') as ax:
            #     ax.tippett(predictions[col].array, predictions['y'].array)

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
        res_df = res_df.reindex(['mfw', 'sv', 'multi', 'comb_a', 'comb_b', 'feat', 'biva'])
        res_df = res_df.reset_index()


# ----------------------------------------------------------------------------------------------------------------------
st.set_page_config(layout="wide")

st.title('Fusion of a speaker and a author verification system - Results exploration')
st.markdown("""---""")

st.header('Results')

# ---------
st.subheader('Performance metrics')
st.write(res_df)
st.write(' ')
st.write(' ')
st.pyplot(sns.catplot(x="method", y="value", col='metric', data=metrics_per_repeat, kind="box"))

# ---------
st.subheader('Error analysis')

# , orient="h"
env_loc_for_plot, table_env_loc = metrics_per_category(['spk_sx', 'spk_dl'], predictions)

# loc_env_chart = alt.Chart(env_loc_for_plot[(env_loc_for_plot.method != 'mfw') & (env_loc_for_plot.metric != 'eer')]).\
#     mark_bar().encode(
#     x="value",
#     y=alt.Y("method", sort=['mfw', 'sv', 'multi', 'comb_a', 'comb_b', 'feat', 'biva']),
#     color=alt.Color("method", scale=alt.Scale(scheme='tableau10'), sort=['mfw', 'sv', 'multi', 'comb_a', 'comb_b',
#                                                                          'feat', 'biva']),
#     column="spk_environments",
#     row='spk_locations'
# )
#
# st.altair_chart(loc_env_chart)
st.write(table_env_loc)

words_for_plot, table_words = metrics_per_category(['spk_num_words'], predictions)
words_diff_for_plot, table_words_diff = metrics_per_category(['words_diff_range'], predictions)

print('hello')

# words_chart = alt.Chart(words_for_plot[(words_for_plot.method != 'mfw') & (words_for_plot.metric != 'eer')]).\
#     mark_bar().encode(
#     x="value",
#     y=alt.Y("method", sort=['mfw', 'sv', 'multi', 'comb_a', 'comb_b', 'feat', 'biva']),
#     color=alt.Color("method", scale=alt.Scale(scheme='tableau10'), sort=['mfw', 'sv', 'multi', 'comb_a', 'comb_b',
#                                                                          'feat', 'biva']),
#     row="spk_num_words"
# )

# words_chart_diff = alt.Chart(words_diff_for_plot[(words_diff_for_plot.method != 'mfw') &
#                                                  (words_diff_for_plot.metric != 'eer')]).mark_bar().encode(
#     x="value",
#     y=alt.Y("method", sort=['mfw', 'sv', 'multi', 'comb_a', 'comb_b', 'feat', 'biva']),
#     color=alt.Color("method", scale=alt.Scale(scheme='tableau10'), sort=['mfw', 'sv', 'multi', 'comb_a', 'comb_b',
#                                                                          'feat', 'biva']),
#     row="words_diff_range"
# )
# cols = st.columns(2)
# cols[0].altair_chart(words_chart)
# cols[1].altair_chart(words_chart_diff)


st.write(table_words)
st.write(table_words_diff)
