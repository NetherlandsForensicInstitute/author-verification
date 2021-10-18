
import scipy.stats
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin


class GaussianCdfTransformer(TransformerMixin):
    def fit(self, X):
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)

        self._valid_features = self._std > 0
        self._mean = self._mean[self._valid_features]
        self._std = self._std[self._valid_features]

        return self

    def transform(self, X):
        assert len(X.shape) == 2
        X = X[:, self._valid_features]
        return scipy.stats.norm.pdf(X, self._mean, self._std)


words = pd.read_csv('output\\model\\fisher_vs_roxsd_600.csv')
words = words[(words['roxsd_valid']) & (words['filler_word'] == 0) & (words['order'] <= 100)]
words = words['word'].tolist()

X_fisher = pd.read_csv("output\\model\\X_fisher_pairs.csv", header=None)
X_fisher.columns = words
y_fisher = pd.read_csv("output\\model\\y_fisher.csv", header=None)
y_fisher.columns = ['label']

X_roxsd = pd.read_csv("output\\model\\X_roxsd_pairs.csv", header=None)
X_roxsd.columns = words
y_roxsd = pd.read_csv("output\\model\\y_roxsd.csv", header=None)
y_roxsd.columns = ['label']

speakers_roxsd = pd.read_csv("output\\model\\roxsd_speakers.csv")
roxsd_metadata = pd.read_csv("output\\model\\roxsd_metadata.csv")
speakers_roxsd = speakers_roxsd.merge(roxsd_metadata, on='spk')
pairs_roxsd = pd.read_csv("output\\model\\roxsd_conv_pairs.csv", header=None)
pairs_roxsd.columns = ['conv1', 'conv2']
lrs_roxsd = pd.read_csv("output\\model\\roxsd_lrs.csv", header=None)
lrs_roxsd.columns = ['lrs']

res_df = pairs_roxsd.join(lrs_roxsd)
res_df = res_df.join(y_roxsd)
speakers_roxsd.columns = speakers_roxsd.columns + '1'
res_df = res_df.merge(speakers_roxsd, left_on='conv1', right_on='convid_side1').drop('convid_side1', axis=1)
speakers_roxsd.columns = [x[:-1] + '2' for x in speakers_roxsd.columns]
res_df = res_df.merge(speakers_roxsd, left_on='conv2', right_on='convid_side2').drop('convid_side2', axis=1)
res_df['correct_pred'] = np.where(((res_df['lrs'] > 1) & (res_df['label'] == 1)) |
                                  ((res_df['lrs'] <= 1) & (res_df['label'] == 0)), 1, 0)
res_df['same_language'] = np.where(res_df['native_language1'] == res_df['native_language2'], 1, 0)

st.set_page_config(layout="wide")
st.title('ROXSD vs FISHER')
st.markdown("""---""")

den = GaussianCdfTransformer()


def values_to_gaussian_pdf(i):

    results = {}
    for case in [['fisher', 0], ['fisher', 1],['roxsd', 0], ['roxsd', 1]]:
        if case[0] == 'fisher':
            df = X_fisher[y_fisher['label'] == case[1]].to_numpy()
        else:
            df = X_roxsd[y_roxsd['label'] == case[1]].to_numpy()

        x = df[:, i]
        df_fitted = den.fit_transform(df)
        y = df_fitted[:, i]
        label = case[0] + '_' + str(case[1])
        results[label] = [x, y]

    return results


st.header('checking distributions')
option = st.radio('Select number of variables:',
                  ['top 4', 'top 12', 'top 24', 'all (92 in total)'])

if 'all' in option:
    l = int(X_roxsd.shape[1])
else:
    l = int(option.split(" ")[1])


cols = st.columns(4)
for i in range(l):
    res = values_to_gaussian_pdf(i)
    fig, ax = plt.subplots()
    ax.plot(res['fisher_0'][0], res['fisher_0'][1], '.', color='b')
    ax.plot(res['fisher_1'][0], res['fisher_1'][1], '.', color='orange')
    ax.plot(res['roxsd_0'][0], res['roxsd_0'][1], '.', color='c')
    ax.plot(res['roxsd_1'][0], res['roxsd_1'][1], '.', color='coral')
    ax.legend(["fisher diff", "fisher same", "roxsd diff", "roxsd same"])
    plt.title(words[i])
    cols[i%4].pyplot(fig)

st.header('checking predictions')

cols = st.columns(2)
lan_size = res_df.groupby(['native_language1', 'native_language2', 'label']).size().reset_index(name='size')
lan_df = res_df.groupby(['native_language1', 'native_language2', 'label']).\
    apply(lambda x: x['correct_pred'].sum()/len(x)).reset_index(name='correct_perc')
lan_df = lan_df.merge(lan_size, on=['native_language1', 'native_language2', 'label'])
cols[1].dataframe(lan_df)

lan_size1 = res_df.groupby(['same_language', 'label']).size().reset_index(name='size')
lan_df1 = res_df.groupby(['same_language', 'label']).\
    apply(lambda x: x['correct_pred'].sum()/len(x)).reset_index(name='correct_perc')
lan_df1 = lan_df1.merge(lan_size1, on=['same_language', 'label'])
cols[0].dataframe(lan_df1)

# group_to_check = st.radio('Select group:', ['same speaker', 'same speaker - wrong prediction',
#                                             'same speaker - correct prediction', 'diff speaker',
#                                             'diff speaker - wrong prediction', 'diff speaker - correct prediction',
#                                             'all the wrong predictions', 'all the correct predictions', 'all'])
#
# if 'same' in group_to_check:
#     df = res_df[res_df['label'] == 1]
# elif 'diff' in group_to_check:
#     df = res_df[res_df['label'] == 0]
# else:
#     df = res_df
#
# if 'correct' in group_to_check:
#     df = df[df['correct_pred'] == 1]
# elif 'wrong' in group_to_check:
#     df = df[df['correct_pred'] == 0]

# st.write(df)
