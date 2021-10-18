import collections
import json
import logging
import os
import re
import numpy as np
import pandas as pd

from nltk.tokenize import WhitespaceTokenizer
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from boilerplate import fileio

LOG = logging.getLogger(__name__)


class FisherDataSource:
    def __init__(self, data, info, n_frequent_words=50, min_words_in_conv=50):
        self._n_freqwords = n_frequent_words  # number of frequent words
        self._data = data  # data location
        self._info = info  # file that connect a side of a conversation to a specific speaker
        self._min_words_in_conv = min_words_in_conv  # min number of words, a conversation should include
        self._wordlist = []

    def _get_cache_path(self):
        filename_safe = re.sub('[^a-zA-Z0-9_-]', '_', self._data)
        return f'.cache/{filename_safe}.json'

    def get(self):
        os.makedirs('.cache', exist_ok=True)
        speakers_path = self._get_cache_path()
        if os.path.exists(speakers_path):
            LOG.debug(f'using cache file: {speakers_path}')
            speakers_wordlist = load_data(speakers_path)
        else:
            speakers_wordlist = compile_data(self._data, self._info)
            store_data(speakers_path, speakers_wordlist)

        # extract a list of frequent words
        # self._wordlist = [word for word, freq in get_frequent_words(speakers_wordlist, self._n_freqwords)]
        words = pd.read_csv('.\\output\\model\\fisher_vs_roxsd_600.csv')
        # & (words['weighted_mean_diff'] < 0.8)
        words = words[(words['roxsd_valid']) & (words['filler_word'] == 0) &
                      (words['order'] <= self._n_freqwords)]
        self._wordlist = words['word'].tolist()

        # build a dictionary of feature vectors
        speakers_conv = filter_speakers_text(speakers_wordlist, self._wordlist, self._min_words_in_conv)

        # convert to X, y
        X, y = to_vector_size(speakers_conv)

        return X, y

    def __repr__(self):
        return f'data(freqwords={self._n_freqwords})'

    @property
    def wordlist(self):
        return self._wordlist


class CreateFeatureVector:
    def __init__(self, wordlist):
        self.wordlist = wordlist

    def __call__(self, texts):

        samples = [[word for word in texts if word in self.wordlist]]

        if len(samples) == 0:
            return []
        else:
            vectorizer = TfidfVectorizer(analyzer='word', use_idf=False, norm=None, vocabulary=self.wordlist,
                                         tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
            return vectorizer.fit_transform(samples).toarray()


def load_data(path):
    with open(path) as f:
        return json.loads(f.read())


def store_data(path, speakers):
    with open(path, 'w') as f:
        f.write(json.dumps(speakers))


def read_session(lines):
    """
    it returns a list of words within the file
    :param lines: <class '_io.TextIOWrapper'>
    """
    lines_to_words = lines.read()
    lines_to_words = re.sub('[0-9]*\.[0-9]*\ [0-9]*\.[0-9]*\ [AB]:', '', lines_to_words)
    lines_to_words = re.sub('\(\(\s+\)\)', '', lines_to_words)  # -> <UNK>
    lines_to_words = re.sub('\(\([a-zUNK0-9\[\]\s\'\-_.,<>]*\)\)', '', lines_to_words)  # -> <GUESS>
    lines_to_words = re.sub('\[\[[a-z]*\]\]', '', lines_to_words)  # -> <SKIP>
    lines_to_words = re.sub('\[[a-z]*\]', '', lines_to_words)  # -> <SOUND>
    lines_to_words = lines_to_words.replace('\n', ' ')

    tk = WhitespaceTokenizer()
    words = tk.tokenize(lines_to_words)

    words = [re.sub(r'^\'', '', i) for i in words]  # remove ' from the beginning of a word
    words = [re.sub(r'\'$', '', i) for i in words]  # remove ' from the end of a word

    return words


def compile_data(index_path, info_path):
    basedir = os.path.dirname(index_path)
    speakers_conv = collections.defaultdict(list)  # create empty dictionary list

    to_ids = {}
    with open(info_path) as f:
        next(f)
        for line in f:
            filename, spk_id, trans = line.split('\t', 2)
            to_ids[filename.replace('-', '_')] = [spk_id.replace('FISHE', ''), trans.replace('\n', '')
                .replace('/WordWave', '')]

    for filepath, digest in tqdm(list(fileio.load_hashtable(index_path).items()), desc='compiling data'):

        # there is some inconsistency with '-' and '_' between actual file names and in the info file
        path_str = os.path.basename(filepath).replace('.txt', '').replace('-', '_')
        spk_conv_id = to_ids.get(path_str)[0] + '_in_' + path_str.replace('fe_03_', '').replace('_a', 'a'). \
            replace('_b', 'b') + '_by_' + to_ids.get(path_str)[1]  # spkid_in_convids_by_transcriber

        with fileio.sha256_open(os.path.join(basedir, filepath), digest, on_mismatch='warn') as f:
            texts = read_session(f)
            if len(texts) > 0:  # ignore 'empty' files
                speakers_conv[spk_conv_id].extend(texts)

    return speakers_conv


def get_data(path, n_frequent_words):
    ds = FisherDataSource(path, n_frequent_words=n_frequent_words)
    return ds.get()


def get_frequent_words(speakers, n):
    """
    returns a dict with the n most frequent words within the conversations transcribed by LDC excluding incomplete words.

    :param speakers: dataset of speakers with the words they used
    :param n: int how many most frequent words will the output contain
    """
    freq_bbn = collections.defaultdict(int)
    freq_ldc = collections.defaultdict(int)
    for sp, sp_words in speakers.items():
        for word in sp_words:
            if not re.compile("[a-z].*-$").match(word):  # exclude incomplete words
                if bool(re.search('BBN', sp)):
                    freq_bbn[word] += 1
                else:
                    freq_ldc[word] += 1
            else:
                continue
    freq_bbn = sorted(freq_bbn.items(), key=lambda x: x[1], reverse=True)
    freq_ldc = sorted(freq_ldc.items(), key=lambda x: x[1], reverse=True)

    word_bbn = [i[0] for i in freq_bbn[:n]]
    mfw = [item for item in freq_ldc[:n] if item[0] in word_bbn]

    return mfw


def filter_speakers_text(speakerdict, wordlist, min_words_in_conv):
    """
    it returns dictionary, each key is a conversation and its values are the relative counts of the most freq words
    if the speaker appears once or they have less words than min_words_in_conv then they are excluded from the analysis

    :param speakerdict: dict of all words used per speaker
    :param wordlist: the n most freq words in corpus
    """

    f = CreateFeatureVector(wordlist)

    # keep speakers with 2 or more conversations
    spk_ids_all = [k.split('_')[0] for k in speakerdict.keys()]  # keep only speaker id
    spk_with_occurrences = [v for v in np.unique(spk_ids_all) if spk_ids_all.count(v) > 1]

    filtered = {}
    for label, texts in speakerdict.items():
        LOG.debug('filter in subset {}'.format(label))

        n_words = len(texts)
        if label.split('_')[0] not in spk_with_occurrences or n_words < min_words_in_conv:
            continue
        else:
            texts = list(f(texts))
            filtered[label] = [100 * i / n_words for i in texts]

    return filtered


def to_vector_size(speakers):
    """
    returns a matrix where each row correspond to a conversation by a speaker, and vector that holds the id of the speaker

    :param speakers: the output of filter_texts_size
    """
    labels = []
    features = []
    for label, texts in speakers.items():
        labels.append(label.split('_')[0])
        features.append(texts)

    return np.concatenate(features), np.array(labels)


# import confidence
#
# if __name__ == '__main__':
#     cfg = confidence.load_name('..\\authorship', 'local')
#     prova = FisherDataSource(cfg.fisher_data, cfg.fisher_data_info, n_frequent_words=5)
#     X, y = prova.get()
#     print(type('3'))
