import numpy as np
import json
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def percentiles_to_string(arr):
    temp = np.percentile(arr, range(0, 101, 10))
    to_print = ''
    for i in range(0, 11):
        if i == 5:
            to_print = to_print + '-- ' + str(round(temp[i], 1)) + ' --, '
        elif i == 10:
            to_print = to_print + str(round(temp[i], 1))
        else:
            to_print = to_print + str(round(temp[i], 1)) + ', '

    st.write('percentiles: [min, 10%, 20%, 30%, 40%, --median--, 60%, 70%, 80%, 90%, max] = [' + to_print + ']')


st.set_page_config(layout="wide")
st.title('ROXSD (non-target) data')
st.markdown("""---""")

st.subheader('Notes on the transcriptions:')
st.markdown("""
- Capital letters are used for names, start of a sentence and for the words I, OK, PM. 
- Symbols appearing in the transcriptions:
    - dot(s) (.) for ending a sentence, and (..) or (...) for pauses or hesitation (maybe also for incomplete words).
    - comma (,) for natural pauses within a sentence. 
    - question mark (?) for questions.
    - explanation mark (!) for excitement.
    - apostrophe (') for contractions (e.g., i'm, i've). They do not seem to be consisted, one can find, for example,
    i'll and i will. Unclear, if this is based on what the speaker said or the transcriber's style. There are cases
    where acute accent (´) has been used instead.
    - dash (-) for split words and as a hesitation marker
- The notation (text) used for:
    - sounds, e.g., haha, 
    - description of a sound, e.g., laughter, 
    - filler words, e.g., yeah
    - unclear utterances
""")
# data_path = 'roxsd/roxsd_clean_for_streamlit.txt'
data_path = 'roxsd/roxsd_for_streamlit.txt'

with open(data_path) as f:
    data = json.load(f)

with open('./output/model/n_freq_words=200_pop_gauss_diff_bray_clf_mlp/wordlist.json', encoding='utf-8') as json_file:
    wordlist = json.load(json_file)

ids_counts = np.array(data['ids_counts'])
num_tokens = np.array(data['num_tokens'])
num_words = np.array(data['num_words'])

within_par = np.array(data['within_par'])
within_par_to_check = data['within_par_to_check']
incomplete = np.array(data['incomplete'])
incomplete_to_check = data['incomplete_to_check']
splits = np.array(data['splits'])
splits_to_check = data['splits_to_check']
other = np.array(data['other'])
other_to_check = data['other_to_check']

freq = data['freq']

st.write('Number of files (2 * num of conversations) = ', len(num_tokens))
st.write('Number of speakers = ', sum(ids_counts))

st.write(pd.DataFrame({
    '# of utterances per spk': [i+1 for i in range(len(ids_counts))],
    '# of speakers': ids_counts,
}))


st.header('Number of tokens')
fig1, ax1 = plt.subplots()
ax1.hist(num_tokens, bins=20)
ax1.set_title('All tokens')
ax1.set_ylabel('counts')
ax1.set_xlabel('# of tokens')

fig2, ax2 = plt.subplots()
ax2.hist(num_words, bins=20)
ax2.set_title('Excluding tokens within parentheses')
ax2.set_ylabel('counts')
ax2.set_xlabel('# of tokens')

col1, col2 = st.columns(2)
col1.pyplot(fig1)
col2.pyplot(fig2)

st.write('For all tokens:')
percentiles_to_string(num_tokens)
st.write('Excluding tokens within parentheses')
percentiles_to_string(num_words)


st.header('Number of words within parentheses')
cols = st.columns(2)
cols[0].write('Files with words within parentheses: ')
input_par = cols[0].text_input('Filter on given word within parentheses (optional): ')
par_df = pd.DataFrame.from_dict(within_par_to_check, orient='index', columns=['instances'])
if input_par == '':
    cols[0].dataframe(par_df)
else:
    cols[0].dataframe(par_df[par_df['instances'].str.contains(input_par)])

cols[1].write('Words within parentheses:')
cols[1].write('')
cols[1].write('')
cols[1].write(within_par)



st.header('Number of split words')

cols = st.columns(2)
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

st.header('Tokens that include symbols not mentioned in the notes')

cols = st.columns(2)
cols[0].write('Files with such tokens: ')
input_token = cols[0].text_input('Filter on given token (optional): ')
other_df = pd.DataFrame.from_dict(other_to_check, orient='index', columns=['instances'])
if input_token == '':
    cols[0].dataframe(other_df)
else:
    cols[0].dataframe(other_df[other_df['instances'].str.contains(input_token)])

cols[1].write('Such tokens:')
cols[1].write('')
cols[1].write('')
cols[1].write(other)


st.header('Frequent words')

freq_df = pd.DataFrame.from_dict(freq, orient='index', columns=['counts']).sort_values('counts', ascending=False).reset_index()
freq_df.rename(columns={'index': 'word'}, inplace=True)
freq_df_top = freq_df.head(500)
mfw_fisher = pd.DataFrame({'word': wordlist})

input_word = st.text_input('Filter on given word (optional): ')
cols = st.columns(2)
cols[0].write('all the words in roxsd')
cols[1].write('the 500 most frequent words in FISHER')
if input_word == '':
    cols[0].dataframe(freq_df)
    cols[1].dataframe(mfw_fisher)
else:
    cols[0].dataframe(freq_df[freq_df['word'].str.contains(input_word)])
    cols[1].dataframe(mfw_fisher[mfw_fisher['word'].str.contains(input_word)])


roxsd_all = freq_df.word.tolist()
roxsd_top500 = freq_df_top.word.tolist()
fisher = mfw_fisher.word.tolist()
diff1 = [w for w in roxsd_top500 if w not in fisher]
diff2 = [w for w in fisher if w not in roxsd_all]

st.write('')
st.write('There are ', len(diff1), ' words in the top 500 roxsd words that are not in the fisher mfw list')
st.write('There are ', len(diff2), ' words in the fisher mfw list that are not in the roxsd')
if len(diff1) + len(diff2) > 0:
    cols = st.columns(2)
    cols[0].write('Words that in the top 500 roxsd words but not in fisher: ')
    cols[0].write(diff1)
    cols[1].write('Words that in fisher list but not in roxsd: ')
    cols[1].write(diff2)


st.subheader('Decisions:')
st.markdown("""
- Everything was converted to lowercase.
- Dots, commas, question marks, explanation marks were removed for this analysis.
- Apostrophes and dashes (') remained as is. Acute accent (´) has been replaced with an apostrophe.
- To match the spelling in FISHER, the following words were replaced:
    - "ok" --> "okay" 
    - "cannot" --> "can\'t"
    - "thats" --> "that\'s'
    - "y[ae]{1,3}[h]{0,1}" --> "yeah"
    - "u[g]{0,1}h" --> "uh" 
    - "u[hm]{0,1}m" --> " um"
    - "aha[h]{0,1}" --> "uh-huh"
    - "oh[o]{0,2}" --> "oh"
    - "hm[m]{0,1}" --> "hm"
    - "ah[h]{0,1}" --> "ah"
    - "mm" --> "mm"
- Anything within parethenses that does not match any pattern above were removed, e.g., (unclear utterance), (laughter),
(haha), (ha), (a)
- (any leftover) parentheses were replaced with space to avoid different words merged to one, e.g, "to(um)" changed 
to "to um "
""")
