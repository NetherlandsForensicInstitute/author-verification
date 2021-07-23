import numpy as np
import json
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

path = 'fisher/data_for_streamlit.txt'
with open(path) as f:
    data = json.load(f)

transcriber = np.array(data['transcriber'])
ids_counts = np.array(data['ids_counts'])
ldc_ids_counts = np.array(data['ldc_ids_counts'])
bbn_ids_counts = np.array(data['bbn_ids_counts'])

num_tokens = np.array(data['num_tokens'])
no_tokens = np.array(data['no_tokens'])
num_words = np.array(data['num_words'])
no_words = np.array(data['no_words'])

num_unk = np.array(data['num_unk'])
num_guess = np.array(data['num_guess'])
num_sounds = np.array(data['num_sounds'])
num_skips = np.array(data['num_skips'])

num_incomplete = np.array(data['num_incomplete'])
incomplete = np.array(data['incomplete'])
incomplete_to_check = data['incomplete_to_check']
num_splits = np.array(data['num_splits'])
splits = np.array(data['splits'])
splits_to_check = data['splits_to_check']
miss_to_check = np.array(data['miss_to_check'])
other = np.array(data['other'])
other_to_check = data['other_to_check']

freq = data['freq']
freq_ldc = data['freq_ldc']
freq_bbn = data['freq_bbn']
freq_bi = data['freq_bi']


st.set_page_config(layout="wide")
st.title('FISHER data')
st.markdown("""---""")

st.write('Number of conversations = ', len(transcriber))

st.write('Number of speakers = ', sum(np.array([4473, 3728, 3636, 115, 14, 4, 1])))
st.write('Number of speakers with more than 1 utterance = ', sum(np.array([3728, 3636, 115, 14, 4, 1])))

st.write(pd.DataFrame({
    '# of utterances per spk': [1, 2, 3, 4, 5, 6, 7],
    '# of speakers': ids_counts,
    '# of speakers in ldc': ldc_ids_counts,
    '# of speakers in bbn': bbn_ids_counts
}))

st.write('Number of conversations transcribed by LDC: ', sum(transcriber == 'ldc') / 2)
st.write('Number of conversations transcribed by BBN: ', sum(transcriber == 'bbn') / 2)

st.subheader('Notes on the transcriptions:')
st.markdown("""
- no capitals (with a few exceptions)
- e.g., L.A. (Los Angeles) is transcribed as: l._a.
- letters are transcribed as the letter followed by a dot, e.g., www is given as w. w. w. and the spelling of store as
s. t. o. r. e.
- (( $~$ )) unclear utterance, mainly used in ldc --> <UNK>
- (( $~$text$~$ )) best estimate by transcriber for LDC/when manual and automatic transcriptions differed significantly
for BBN --> <GUESS>
- [type of noise] for noises such as laughter, mostly found in bbn --> <SOUND>
- the conversations are mostly in English, but one might spot other languages as well, e.g., fe_03_00265-(ab) includes
French
- in 8 files (e.g., fe_03_00041-a, fe_03_00082-(ab), fe_03_00238-a), all transcribed by LDC, one can spot: '[[skip]]'
--> <SKIP>
- Apostrophe (') seems to be used for the following cases:
    - for contractions (i'm, aren't, there's)
    - to mention a specific word, e.g., 'absolutely' in fe_03_01296-a [... crazy with the word 'absolutely']
    - to refer to a phrase or quote someone, e.g., in fe_03_07334-b [... on the other line 'here's how you can fix it'],
    and in fe_03_08839-b [... or whomever say 'here it comes']
    - to denote a half spoken word, e.g, 'xactly in fe_03_08910-b and 'member in fe_03_05739-a. BBN used extensively to
    denote 'cause while there is no such instance in LDC
- Dash (-) used for the following:
    - for split words such as 'father-in-law', 'make-up', 'no-one', and 'bye-bye' (not clear if the transcription of
    such words is consistent throughout)
    - as hesitation marks e.g., y-you and i-i (which are included in the split word list)
    - for incomplete words, e.g., par- and sophis-
    - some times it looks that it is used also for "incomplete" sentences (e.g., '... each-' in fe_03_01463-b,
    '... actually-' in fe_03_04706, and '... personal-' in fe_03_06746)
- numbers are written out, e.g., seventy-five in fe_03_03524-b (with a few exceptions)

""")


def plot_this(arr, xlimits=None, ylimits=None):
    col1, col2, col3 = st.beta_columns(3)

    fig1, ax1 = plt.subplots()
    ax1.hist(arr, bins=20)
    ax1.set_title('All')
    ax1.set_ylabel('counts')

    fig2, ax2 = plt.subplots()
    ax2.hist(arr[transcriber == 'ldc'], bins=20)
    ax2.set_title('by LDC')
    ax2.set_xlabel('# of tokens')

    fig3, ax3 = plt.subplots()
    ax3.hist(arr[transcriber == 'bbn'], bins=20)
    ax3.set_title('by BBN')

    if xlimits is not None and ylimits is not None:
        ax1.set(xlim=xlimits, ylim=ylimits)
        ax2.set(xlim=xlimits, ylim=ylimits)
        ax3.set(xlim=xlimits, ylim=ylimits)

    col1.pyplot(fig1)
    col2.pyplot(fig2)
    col3.pyplot(fig3)


def percentiles_to_string(arr):
    temp = np.percentile(arr, range(0, 101, 10))
    to_print = str(temp[0]) + ', ' + str(temp[1]) + ', ' + str(temp[2]) + ', ' + str(temp[3]) + ', ' + str(temp[4]) + \
               ', --' + str(temp[5]) + '--, ' + str(temp[6]) + ', ' + str(temp[7]) + ', ' + str(temp[8]) + ', ' \
               + str(temp[9]) + ', ' + str(temp[10])
    st.write('percentiles: [min, 10%, 20%, 30%, 40%, --median--, 60%, 70%, 80%, 90%, max] = [' + to_print + ']')


def avg_to_string(arr):
    st.write('avg = ', np.round(np.mean(arr), 2), ' || avg (LDC only) = ', np.round(np.mean(arr[transcriber == 'ldc']), 2),
             ' || avg (BBN only) = ', np.round(np.mean(arr[transcriber == 'bbn']), 2))
    st.write('')


st.header('Number of tokens')
st.write('Number of tokens INCLUDING incomplete words, <UNK>, <GUESS>, <SOUND>, <SKIP>')
percentiles_to_string(num_tokens)
plot_this(num_tokens, xlimits=(0, 2500), ylimits=(0, 4000))

st.write('Number of tokens EXCLUDING incomplete words, <UNK>, <GUESS>, <SOUND>, <SKIP>')
percentiles_to_string(num_words)
plot_this(num_words, xlimits=(0, 2500), ylimits=(0, 4000))


st.write('There are ', len(no_tokens), ' files with less than 20 tokens and ', len(no_words), ' files with less than 50 words')

cols = st.beta_columns(2)
cols[0].write('Files with few tokens: ')
cols[0].write(no_tokens)
cols[1].write('Files with few words: ')
cols[1].write(no_words)

st.header('Number of <UNK>')
avg_to_string(num_unk)
plot_this(num_unk)

st.header('Number of <GUESS>')
avg_to_string(num_guess)
plot_this(num_guess)

st.header('Number of <SOUND>')
avg_to_string(num_sounds)
plot_this(num_sounds)

st.header('Number of <SKIP>')
st.write('num of files including <SKIP> = ', np.sum(num_skips > 0), ' || num of files including <SKIP> (LDC only) = ',
         np.sum(num_skips[transcriber == 'ldc'] > 0), ' || num of files including <SKIP> (BBN only) = ',
         np.sum(num_skips[transcriber == 'bbn'] > 0))

st.header('Missed (( ... )) and [[ ... ]]')
st.write('Number of missed \'((\', \'))\', \'[[\', or \']]\' = ', len(miss_to_check))
if len(miss_to_check) > 0:
    st.write('Files to check', miss_to_check)

st.header('Number of incomplete words')
avg_to_string(num_incomplete)

cols = st.beta_columns(2)
cols[0].write('Files with incomplete words: ')
input_incomplete = cols[0].text_input('Filter on given incomplete word (optional): ')
incomplete_df = pd.DataFrame.from_dict(incomplete_to_check, orient='index', columns=['instances'])
if input_incomplete == '':
    cols[0].dataframe(incomplete_df)
else:
    cols[0].dataframe(incomplete_df[incomplete_df['instances'].str.contains(input_incomplete)])

cols[1].write('Incomplete words:')
cols[1].write('')
cols[1].write('')
cols[1].write(incomplete)

st.header('Number of split words')
avg_to_string(num_splits)

cols = st.beta_columns(2)
cols[0].write('Files with split words: ')
input_split = cols[0].text_input('Filter on given split word (optional): ')
split_df = pd.DataFrame.from_dict(splits_to_check, orient='index', columns=['instances'])
if input_split == '':
    cols[0].dataframe(split_df)
else:
    cols[0].dataframe(split_df[split_df['instances'].str.contains(input_split)])

cols[1].write('Split words:')
cols[1].write('')
cols[1].write('')
cols[1].write(splits)

st.header('Tokens that include other symbols than letters, -, <, and >')
st.write('Number of files with such tokens', len(other_to_check.keys()))
cols = st.beta_columns(2)
cols[0].write('Files with such tokens: ')
input_token = cols[0].text_input('Filter on given token (optional): ')
other_df = pd.DataFrame.from_dict(other_to_check, orient='index', columns=['instances'])
if input_split == '':
    cols[0].dataframe(other_df)
else:
    cols[0].dataframe(other_df[other_df['instances'].str.contains(input_token)])

cols[1].write('Such tokens:')
cols[1].write('')
cols[1].write('')
cols[1].write(other)

st.header('Unigrams and bigrams')
st.subheader('Unigrams')
wc = WordCloud(width=2500, height=400)
wc.generate_from_frequencies(frequencies=freq)
st.image(wc.to_array())

st.subheader('Bigrams')
wc2 = WordCloud(width=2500, height=400)
wc2.generate_from_frequencies(frequencies=freq_bi)
st.image(wc2.to_array())

st.header('Most frequent words and bigrams')


def most_frequent(freq):

    df = pd.DataFrame.from_dict(freq, orient='index', columns=['counts']).sort_values('counts', ascending=False)
    num = []
    perc = []
    mfw = 100*(df.cumsum() / df.sum())
    mfw = mfw.round()

    for i in range(10, 110, 10):
        mfw_subset = mfw[mfw['counts'] <= i]
        num.append(mfw_subset.shape[0])

    for j in np.array([10, 50, 100, 150, 200, 300, 400, 500, 600, 1000]):
        mfw_subset = mfw.head(j)
        perc.append(mfw_subset.iat[-1, 0])

    return df, num, perc


freq_df, num_mfw, perc_mfw = most_frequent(freq)
freq_ldc_df, num_mfw_ldc, perc_mfw_ldc = most_frequent(freq_ldc)
freq_bbn_df, num_mfw_bbn, perc_mfw_bbn = most_frequent(freq_bbn)
freq_bi_df, num_mfbi, perc_mfbi = most_frequent(freq_bi)
mf_df = pd.DataFrame({'perc': range(10, 110, 10), 'freq words': num_mfw, 'freq words (ldc)': num_mfw_ldc,
                      'freq words (bbn)': num_mfw_bbn, 'freq bigrams': num_mfbi})
mf_df_rev = pd.DataFrame({'num of top words/bigrams': np.array([10, 50, 100, 150, 200, 300, 400, 500, 600, 1000]),
                          'perc (words)': perc_mfw, 'perc (bigrams)': perc_mfbi})


cols = st.beta_columns(2)
cols[0].dataframe(mf_df)
cols[1].dataframe(mf_df_rev)

input_top = st.number_input('Most frequent words/bigrams to display (max 1000)', 10, 1000, 10)
freq_df_top = freq_df.head(input_top)
freq_ldc_df_top = freq_ldc_df.head(input_top)
freq_bbn_df_top = freq_bbn_df.head(input_top)
freq_df_top = freq_df.head(input_top)
cols = st.beta_columns(4)
cols[0].write('Frequent words')
cols[0].dataframe(freq_df_top)
cols[1].write('Frequent words in ldc')
cols[1].dataframe(freq_ldc_df_top)
cols[2].write('Frequent words in bbn')
cols[2].dataframe(freq_bbn_df_top)
cols[3].write('Frequent bigrams')
cols[3].dataframe(freq_bi_df.head(input_top))

all = freq_df_top.index.tolist()
ldc = freq_ldc_df_top.index.tolist()
bbn = freq_bbn_df_top.index.tolist()
diff1 = list(set(ldc) - set(bbn))
diff2 = list(set(bbn) - set(ldc))

st.write('')
st.write('There are ', len(list(set(ldc) - set(bbn)) + list(set(bbn) - set(ldc))),
         ' differences between the ldc list and the bbn list')
if len(diff1) + len(diff1) > 0:
    cols = st.beta_columns(2)
    cols[0].write('Words that in ldc list but not in bbn: ')
    cols[0].write(diff1)
    cols[1].write('Words that in bbn list but not in ldc: ')
    cols[1].write(diff2)

st.write('')
st.write('')
st.write('Split words and words with other symbols than letters, -, <, and > in the most frequent words list')


def find_ls1_elem_also_in_ls2(ls1, ls2):
    elem = [item for item in ls1 if item in ls2]
    return elem


selection = st.selectbox('List of most frequent words are based on:', ['all', 'ldc', 'bbn'])
if selection == 'ldc':
    mf_list = ldc
elif selection == 'bbn':
    mf_list = bbn
else:
    mf_list = all

split_in_fw = find_ls1_elem_also_in_ls2(mf_list, list(splits))
other_in_fw = find_ls1_elem_also_in_ls2(mf_list, list(other))

st.write('There are ', len(split_in_fw), ' split words in the list')
st.write('There are ', len(other_in_fw), ' words with other symbols than letters, -, <, and > in the list')

cols = st.beta_columns(2)
if len(split_in_fw) > 0:
    cols[0].write('Split-words in the frequent words:')
    cols[0].write(split_in_fw)
if len(other_in_fw) > 0:
    cols[1].write('Words with other symbols than letters, -, <, and > in the frequent words:')
    cols[1].write(other_in_fw)

st.header('Decisions')
st.markdown("""
- <UNK>, <GUESS>, <SOUND>, <SKIP> are excluded from deriving the most frequent word list and from counting the number of
words per speaker per conversation
- incomplete words are excluded from deriving the most frequent word list but not from counting number of
words per speaker per conversation
- as a starting point no extra actions for contractions or general for handling apostrophes or dashes
- similar split words and non-english words stay as they are
- again as a starting point, only conversations from LDC are taken into account in the analysis
""")