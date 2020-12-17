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
    def __init__(self, data, n_frequent_words, tokens_per_sample):
        self._n_freqwords = n_frequent_words
        self._tokens_per_sample = tokens_per_sample
        self._data = data

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
            speakers_wordlist = compile_data(self._data)
            store_data(speakers_path, speakers_wordlist)

        # extract a list of frequent words
        wordlist = [word for word, freq in get_frequent_words(speakers_wordlist, self._n_freqwords)]

        # build a dictionary of feature vectors
        speakers = filter_texts_size(speakers_wordlist, wordlist, self._tokens_per_sample)

        # convert to X, y
        X, y = to_vector_size(speakers)

        return X, y

    def __repr__(self):
        return f'data(freqwords={self._n_freqwords}; ntokens={self._tokens_per_sample})'


class ExtractWords:
    def __call__(self, texts):
        yield from texts


class RearrangeSamples:
    def __init__(self, n, wordlist):
        self.n = n
        self.wordlist = wordlist

    def __call__(self, items):
        for i in range(len(items) // self.n):
            yield [word for word in items[i*self.n: (i+1)*self.n] if word in self.wordlist]


class CreateFeatureVector:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, samples):
        if len(samples) == 0:
            return []
        else:
            vectorizer = TfidfVectorizer(analyzer='word', use_idf=False, norm=None, vocabulary=self.vocabulary,
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

    remember:
    *v: non-Dutch  words,  *n:  new  non-existing  words,  *s:  street  words,
    *a: incomplete  words, *u:  distorted  words, *x: unclear word
    we keep the words with their notation (for *n, *a, *u, and *x) to be able to exclude them when we
    derive the set of the most frequent words
    """
    lines_to_words = lines.read()
    lines_to_words = re.sub('[0-9]*\.[0-9]*\t', '', lines_to_words)

    lines_to_words = lines_to_words.replace('start\tend\ttext\n', '').replace('.', '').replace('-', ' ')\
        .replace('?', '').replace('\n', ' ').replace('xxx', '').replace('ggg', '').replace('vvv', '')\
        .replace('*v', '').replace('*s', '')
    s = lines_to_words.translate({ord(c): None for c in string.punctuation if c != '*'})
    tk = WhitespaceTokenizer()
    words = tk.tokenize(s)

    return words


def compile_data(index_path):
    basedir = os.path.dirname(index_path)
    speakers = collections.defaultdict(list)  # create empty dictionary list

    for filepath, digest in tqdm(list(fileio.load_hashtable(index_path).items()), desc='compiling data'):
        path_str = os.path.basename(filepath)
        speaker_id = path_str[:len(path_str) - 10]  # basename path
        with fileio.sha256_open(os.path.join(basedir, filepath), digest, on_mismatch='warn') as f:
            texts = read_session(f)
            speakers[speaker_id].extend(texts)

    return speakers


def get_data(path, n_frequent_words, tokens_per_sample):
    ds = DataSource(path, n_frequent_words=n_frequent_words, tokens_per_sample=tokens_per_sample)
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
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return freq[:n]


def filter_texts_size(speakerdict, wordlist, tokens_per_sample):
    """
    it returns per speaker all possible samples (the length of the samples is equal to the lenght of the wordlist and
    its values are the frequency of the words in the wordlist for that speaker/sample).

    :param speakerdict: dict of all words used per speaker
    :param wordlist: the n most freq words in corpus
    :param tokens_per_sample: number of words within a sample
    """
    filters = [
        ExtractWords(),
        RearrangeSamples(tokens_per_sample, wordlist),
        CreateFeatureVector(wordlist),
    ]
    filtered = {}
    for label, texts in speakerdict.items():
        LOG.debug('filter in subset {}'.format(label))
        for f in filters:
            texts = list(f(texts))
        if len(texts) != 0:
            filtered[label] = texts

    return filtered


def to_vector_size(speakers):
    """
    returns a matrix where each row correspond to a sample by a speaker, and vector that holds the id of the speaker

    :param speakers: the output of filter_texts_size
    """
    labels = []
    features = []
    for label, texts in speakers.items():
        speaker_id = label[2:]
        labels.append(len(texts) * [speaker_id])
        features.append(texts)

    return np.concatenate(features), np.concatenate(labels)
