import collections
import json
import logging
import os
import re
import string

from nltk import word_tokenize
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
        wordlist = [ word for word, freq in Function_file.get_frequent_words(speakers_wordlist, self._n_freqwords) ]

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


all_words = 0

def read_session(lines):
    global all_words
    global test
    speakers = []
    test = lines.read()
    test = re.sub('[0-9]*\.[0-9]*\t', '', test)
    test = re.sub('[A-Za-z]*\*n','',test)
    test = re.sub('[A-Za-z]*\*u','',test)
    test = re.sub('[A-Za-z]*\*a','',test)
    test = re.sub('[A-Za-z]*\*x','',test)
    test = test.replace('start\tend\ttext\n', '').replace('.', '').replace('-', ' ').replace('?', '').replace('\n',' ').replace('xxx', '').replace('ggg', '').replace('vvv', '').replace('*v','').replace('*s','')
    s = test
    s = s.translate({ord(c): None for c in string.punctuation})
    all_words += len(word_tokenize(s))
    # print(word_tokenize(s))
    speakers.extend(word_tokenize(s))
    #print('all words in session:', all_words)
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
        speakerid = path_str[:len(path_str) - 10]# basename path
        with fileio.sha256_open(os.path.join(basedir, filepath), digest, on_mismatch='warn') as f:
            texts = read_session(f)
            speakers[speakerid].extend(texts)

    return speakers


def get_data(path, n_frequent_words, tokens_per_sample):
    ds = DataSource(path, n_frequent_words=n_frequent_words, tokens_per_sample=tokens_per_sample)
    return ds.get()
