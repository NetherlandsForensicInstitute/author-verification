import collections
import json
import logging
import os
import re
import numpy as np

from nltk.tokenize import WhitespaceTokenizer
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from boilerplate import fileio

LOG = logging.getLogger(__name__)


class FisherDataSource:
    def __init__(self, data, info, n_frequent_words):
        self._n_freqwords = n_frequent_words
        self._data = data
        self._info = info

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
        wordlist = [word for word, freq in get_frequent_words(speakers_wordlist, self._n_freqwords)]

        # build a dictionary of feature vectors
        speakers_cov = filter_texts_size(speakers_wordlist, wordlist)

        # convert to X, y
        X, y = to_vector_size(speakers_cov)

        return X, y

    def __repr__(self):
        return f'data(freqwords={self._n_freqwords})'


class ExtractWords:
    def __call__(self, texts):
        yield from texts


class RearrangeSamples:
    def __init__(self, wordlist):
        self.wordlist = wordlist

    def __call__(self, items):
        yield [word for word in items if word in self.wordlist]


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
    """
    lines_to_words = lines.read()
    lines_to_words = re.sub('[0-9]*\.[0-9]*\ [0-9]*\.[0-9]*\ [AB]:', '', lines_to_words)
    lines_to_words = re.sub('\(\(\s+\)\)', '<UNK>', lines_to_words)
    lines_to_words = re.sub('\(\([a-zUNK0-9\[\]\s\'\-_.,<>]*\)\)', '<GUESS>', lines_to_words)
    lines_to_words = re.sub('\[\[[a-z]*\]\]', '<SKIP>', lines_to_words)
    lines_to_words = re.sub('\[[a-z]*\]', '<SOUND>', lines_to_words)
    lines_to_words = lines_to_words.replace('\n', ' ')

    tk = WhitespaceTokenizer()
    words = tk.tokenize(lines_to_words)

    return words


def compile_data(index_path, info_path, min_words_in_conv=0):
    basedir = os.path.dirname(index_path)
    speakers_conv = collections.defaultdict(list)  # create empty dictionary list

    to_ids = {}
    with open(info_path) as f:
        next(f)
        for line in f:
            filename, spk_id, trans = line.split('\t', 2)
            to_ids[filename.replace('-', '_')] = [spk_id.replace('FISHE', ''), trans.replace('\n', '').replace('/WordWave', '')]

    for filepath, digest in tqdm(list(fileio.load_hashtable(index_path).items()), desc='compiling data'):

        # there is some inconsistency with '-' and '_' between actual file names and in the info file
        path_str = os.path.basename(filepath).replace('.txt', '').replace('-', '_')
        spk_conv_id = to_ids.get(path_str)[0] + '_in_' + path_str.replace('fe_03_', '').replace('_a', 'a').\
            replace('_b', 'b') + '_by_' + to_ids.get(path_str)[1]

        with fileio.sha256_open(os.path.join(basedir, filepath), digest, on_mismatch='warn') as f:
            texts = read_session(f)
            if len(texts) >= min_words_in_conv:
                speakers_conv[spk_conv_id].extend(texts)

    return speakers_conv


def get_data(path, n_frequent_words):
    ds = FisherDataSource(path, n_frequent_words=n_frequent_words)
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
            if word != '<UNK>' or word != '<GUESS>' or word != '<SOUND>':
                continue
            else:
                freq[word] += 1
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return freq[:n]


def filter_texts_size(speakerdict, wordlist):
    """
    it returns one conversation of one speaker  (the length of the samples is equal to the length of the wordlist and
    its values are the frequency of the words in the wordlist for that speaker/conversation).

    :param speakerdict: dict of all words used per speaker
    :param wordlist: the n most freq words in corpus
    """
    filters = [
        ExtractWords(),
        RearrangeSamples(wordlist),
        CreateFeatureVector(wordlist),
    ]
    filtered = {}
    for label, texts in speakerdict.items():
        LOG.debug('filter in subset {}'.format(label))
        # n_words = len(texts)
        for f in filters:
            texts = list(f(texts))
        if len(texts) != 0:
            filtered[label] = texts  # [100*i/n_words for i in texts]

    return filtered


def to_vector_size(speakers):
    """
    returns a matrix where each row correspond to a conversation by a speaker, and vector that holds the id of the speaker

    :param speakers: the output of filter_texts_size
    """
    labels = []
    features = []
    for label, texts in speakers.items():
        speaker_id = label[2:(len(label) - 2)]
        labels.append(speaker_id)
        features.append(texts)

    return np.concatenate(features), np.array(labels)


import confidence

if __name__ == '__main__':
    cfg = confidence.load_name('..\\authorship', 'local')
    prova = FisherDataSource(cfg.fisher_data, cfg.fisher_data_info, n_frequent_words=200)
    X, y = prova.get()
    print(type('3'))
