import collections
import json
import logging
import os
import re
import string

from nltk.tokenize import WhitespaceTokenizer
from tqdm import tqdm

from boilerplate import fileio
from . import Function_file


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
        wordlist = [word for word, freq in Function_file.get_frequent_words(speakers_wordlist, self._n_freqwords)]

        # build a dictionary of feature vectors
        speakers = Function_file.filter_texts_size_new(speakers_wordlist, wordlist, self._tokens_per_sample)

        # convert to X, y
        X, y = Function_file.to_vector_size(speakers)

        return X, y

    def __repr__(self):
        return f'data(freqwords={self._n_freqwords}; ntokens={self._tokens_per_sample})'


def load_data(path):
    with open(path) as f:
        return json.loads(f.read())


def store_data(path, speakers):
    with open(path, 'w') as f:
        f.write(json.dumps(speakers))


def read_session(lines):
    speakers = []
    test = lines.read()
    test = re.sub('[0-9]*\.[0-9]*\t', '', test)

    # remember:
    # *v: non-Dutch  words,  *n:  new  non-existing  words,  *s:  street  words,
    # *a: incomplete  words, *u:  distorted  words, *x: unclear word
    # we keep the words with their notation (for *n, *a, *u, and *x) to be able to exclude them when we
    # derive the set of the most frequent words
    test = test.replace('start\tend\ttext\n', '').replace('.', '').replace('-', ' ').replace('?', '').replace('\n', ' ') \
        .replace('xxx', '').replace('ggg', '').replace('vvv', '').replace('*v','').replace('*s','')
    s = test.translate({ord(c): None for c in string.punctuation if c != '*'})
    tk = WhitespaceTokenizer()
    conv = tk.tokenize(s)
    if len(conv) > 30:
        speakers.extend(conv)

    return speakers


def read_list(path):
    with open(path) as f:
        for line in f:
            pos = line.find('  ')
            filehash = line[:pos]
            filepath = line[pos + 2:-1]
            yield filehash, filepath


def compile_data(index_path):
    basedir = os.path.dirname(index_path)
    speakers = collections.defaultdict(list)  # create empty dictionary list

    for filepath, digest in tqdm(list(fileio.load_hashtable(index_path).items()), desc='compiling data'):
        path_str = os.path.basename(filepath)
        speaker_id = path_str[:len(path_str) - 10]# basename path
        print(speaker_id)
        with fileio.sha256_open(os.path.join(basedir, filepath), digest, on_mismatch='warn') as f:
            texts = read_session(f)
            speakers[speaker_id].extend(texts)

    return speakers


def get_data(path, n_frequent_words, tokens_per_sample):
    ds = DataSource(path, n_frequent_words=n_frequent_words, tokens_per_sample=tokens_per_sample)
    return ds.get()
