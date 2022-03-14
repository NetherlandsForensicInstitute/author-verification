import collections
import json
import logging
import os
import re
import string
import numpy as np

from nltk.tokenize import WhitespaceTokenizer
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from boilerplate import fileio


LOG = logging.getLogger(__name__)


class DataSource:
    def __init__(self, data, n_frequent_words, min_num_of_words):
        self._n_freqwords = n_frequent_words
        self._data = data
        self._min_num_of_words = min_num_of_words

    def _get_cache_path(self):
        filename_safe = re.sub('[^a-zA-Z0-9_-]', '_', self._data)
        filename_safe = re.sub('_csv', '', filename_safe)
        return f'.cache/{filename_safe}.json'

    def get(self):
        os.makedirs('.cache', exist_ok=True)
        speakers_path = self._get_cache_path()
        if os.path.exists(speakers_path):
            LOG.debug(f'using cache file: {speakers_path}')
            speakers_wordlist = load_data(speakers_path)
        else:
            speakers_wordlist = compile_data(self._data)
            store_data(speakers_path, speakers_wordlist)

        # extract a list of frequent words
        wordlist = [word for word, freq in get_frequent_words(speakers_wordlist, self._n_freqwords)]

        # build a dictionary of feature vectors
        speakers_conv = filter_texts(speakers_wordlist, wordlist, self._min_num_of_words)

        # convert to X, y (= spk id) but keep the conversation id
        X, y, conv_ids = to_vector_size(speakers_conv)

        return X, y, conv_ids

    def __repr__(self):
        return f'data(freqwords={self._n_freqwords})'


class CreateFeatureVector:
    def __init__(self, wordlist):
        self.wordlist = wordlist

    def __call__(self, texts):
        """
        it takes a list of words used in a conversation by one speaker and returns the counts of the words in the
        wordlist based on this conversation
        """
        sample = [[word for word in texts if word in self.wordlist]]

        if len(sample) == 0:
            return []
        else:
            vectorizer = TfidfVectorizer(analyzer='word', use_idf=False, norm=None, vocabulary=self.wordlist,
                                         tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
            return vectorizer.fit_transform(sample).toarray()


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

    remember:
    *v: non-Dutch  words,  *n:  new  non-existing  words,  *s:  street  words,
    *a: incomplete  words, *u:  distorted  words, *x: unclear word
    we keep the words with their notation (for *n, *a, *u, and *x) to be able to exclude them when we
    derive the set of the most frequent words
    """
    lines_to_words = lines.read()
    lines_to_words = re.sub('[0-9]*\.[0-9]*\t', '', lines_to_words)  # to remove timestamps
    lines_to_words = re.sub('[A-Za-z]*\*[anuxANUX]{1}', '', lines_to_words)
    lines_to_words = re.sub('[A-Za-z]*\*[etV]{1}', '', lines_to_words)  # unknown notation
    lines_to_words = re.sub('[A-Za-z]*\*op', '', lines_to_words)  # a mistake?

    lines_to_words = lines_to_words.replace('start\tend\ttext\n', '').replace('.', '').replace('-', ' ')\
        .replace('?', '').replace('\n', ' ').replace('xxx', '').replace('ggg', '').replace('vvv', '')\
        .replace('*v', '').replace('*s', '')

    lines_to_words = re.sub('[A-Za-z]*\*', '', lines_to_words)  # it seems that there are words with missing notation

    # s = lines_to_words.translate({ord(c): None for c in string.punctuation if c != '*'})
    tk = WhitespaceTokenizer()
    words = tk.tokenize(lines_to_words)

    return words


def compile_data(index_path):
    basedir = os.path.dirname(index_path)
    speakers_conv = collections.defaultdict(list)  # create empty dictionary list

    for filepath, digest in tqdm(list(fileio.load_hashtable(index_path).items()), desc='compiling data'):
        path_str = os.path.basename(filepath)
        speaker_conv_id = path_str[:len(path_str) - 6].replace('-', '')  # basename path
        with fileio.sha256_open(os.path.join(basedir, filepath), digest, on_mismatch='warn') as f:
            texts = read_session(f)
            if len(texts) > 0:  # to exclude empty files
                speakers_conv[speaker_conv_id].extend(texts)

    return speakers_conv


def get_data(path, n_frequent_words):
    ds = DataSource(path, n_frequent_words=n_frequent_words)
    return ds.get()


def get_frequent_words(speakers, n):
    """
    returns a dict with the n most frequent words within the provided data excluding words that are non-existing words,
     incomplete, distorted or unclear (noted with *n, *a, *u, and *x in the data).

    :param speakers: dataset of speakers with the words they used
    :param n: int how many most frequent words will the output contain
    """
    freq = collections.defaultdict(int)
    for sp in speakers.values():
        for word in sp:
            if '*' in word:
                continue
            else:
                freq[word] += 1

    with open('frida/predictions/word_frequencies.json', 'w') as fp:
        json.dump(freq, fp)
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return freq[:n]


def filter_texts(speakerdict, wordlist, min_num_of_words):
    """
    it returns a dict that each key correspond to a conversation of one speaker and its value is a vector of length
    equal to the length of the wordlist and its values are the relative frequencies of the words in the wordlist for
    that speaker/conversation).

    :param speakerdict: dict of all words used per speaker in a specific conversation
    :param wordlist: the n most freq words in corpus
    """
    f = CreateFeatureVector(wordlist)

    filtered = {}
    conv_lens = {}
    for label, texts in speakerdict.items():
        LOG.debug('filter in subset {}'.format(label))
        ltexts = len(texts)
        conv_lens[label] = ltexts
        filtered_texts = list(f(texts))
        if ltexts > min_num_of_words:
            filtered[label] = [100*i/ltexts for i in filtered_texts]  # texts

    with open('frida/predictions/conversation_length.json', 'w') as fp:
        json.dump(conv_lens, fp)

    filtered_to_save = {k: val[0].tolist() for k, val in filtered.items()}
    with open('frida/predictions/conversation_relative_frequencies.json', 'w') as fp:
        json.dump(filtered_to_save, fp)

    return filtered


def to_vector_size(speakers):
    """
    returns a matrix where each row correspond to a conversation by a speaker, a vector that holds the id of the
    speaker, and a vector that holds the id of the conversation

    :param speakers: the output of filter_texts_size
    """
    features = []
    speaker_ids = []
    conv_ids = []

    for conv_id, texts in speakers.items():
        features.append(texts)
        speaker_ids.append(conv_id[2:(len(conv_id) - 2)])
        conv_ids.append(conv_id)

    return np.concatenate(features),  np.array(speaker_ids),  np.array(conv_ids)
