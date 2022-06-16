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


# used only to validate a model, not to train one
class RoxsdDataSource:
    def __init__(self, data, dir, min_words_in_conv=50):
        self._data = data  # data location
        self._dir = dir  # location of the freq word list
        self._min_words_in_conv = min_words_in_conv  # min number of words, a conversation should include

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
        with open(f'{self._dir}/wordlist.json', encoding='utf-8') as json_file:
            wordlist = json.load(json_file)

        # build a dictionary of feature vectors
        speakers_conv = filter_speakers_text(speakers_wordlist, wordlist, self._min_words_in_conv)

        # convert to X, y
        X, y, conv_ids = to_vector_size(speakers_conv)

        return X, y, conv_ids

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


def clean_text(lines_to_words):
    lines_to_words = lines_to_words.lower()
    lines_to_words = lines_to_words.replace('\n', ' ')
    lines_to_words = re.sub('[.,?!]', '', lines_to_words)
    lines_to_words = re.sub('\u00B4', '\'', lines_to_words)
    lines_to_words = re.sub('(ok)', ' okay ', lines_to_words)
    lines_to_words = lines_to_words.replace('cannot', 'can\'t').replace('thats', 'that\'s').replace('(mm)', ' mm ')
    lines_to_words = re.sub('\(u[hm]{0,1}m\)', ' um ', lines_to_words)
    lines_to_words = re.sub('\(u[g]{0,1}h\)', ' uh ', lines_to_words)
    lines_to_words = re.sub('\(hm[m]{0,1}\)', ' hm ', lines_to_words)
    lines_to_words = re.sub('\(ah[h]{0,1}\)', ' ah ', lines_to_words)
    lines_to_words = re.sub('\(aha[h]{0,1}\)', ' uh-huh ', lines_to_words)
    lines_to_words = re.sub('\(oh[o]{0,2}\)', ' oh ', lines_to_words)
    lines_to_words = re.sub('\(y[ae]{1,3}[h]{0,1}\)', ' yeah ', lines_to_words)
    lines_to_words = re.sub('\([a-z\s].*\)', '', lines_to_words)
    lines_to_words = lines_to_words.replace('(', ' ').replace(')', ' ')

    return lines_to_words


def compile_data(index_path):
    basedir = os.path.dirname(index_path)
    speakers_conv = collections.defaultdict(list)  # create empty dictionary list

    for filepath, digest in tqdm(list(fileio.load_hashtable(index_path).items()), desc='compiling data'):

        with fileio.sha256_open(os.path.join(basedir, filepath), digest, on_mismatch='warn') as f:

            lines = f.readlines()
            for i, line in enumerate(lines):
                info = re.split(r'\t+', line)
                if len(info) < 5:
                    continue

                if info[3] == 'L':
                    spk_id = info[0] + '_A' + ';' + info[4]  # L and A stands for caller
                else:
                    spk_id = info[0] + '_B' + ';' + info[4]  # R and B stands for receiver

                lines_to_words = clean_text(info[5])

                tk = WhitespaceTokenizer()
                words = tk.tokenize(lines_to_words)
                if len(words) > 0:  # ignore 'empty' files
                    speakers_conv[spk_id].extend(words)

    return speakers_conv


def get_data(path, n_frequent_words):
    ds = RoxsdDataSource(path, n_frequent_words=n_frequent_words)
    return ds.get()


def filter_speakers_text(speakerdict, wordlist, min_words_in_conv):
    """
    it returns dictionary, each key is a conversation and its values are the relative counts of the most freq words
    if the speaker appears once or they have less words than min_words_in_conv then they are excluded from the analysis

    :param speakerdict: dict of all words used per speaker
    :param wordlist: the n most freq words in corpus
    """
    f = CreateFeatureVector(wordlist)

    # keep speakers with 2 or more conversations (not valid)
    spk_ids_all = [k.split(';')[1] for k in speakerdict.keys()]  # keep only speaker id
    spk_with_occurrences = [v for v in np.unique(spk_ids_all) if spk_ids_all.count(v) < 4]
    # print(spk_with_occurrences)

    filtered = {}
    for label, texts in speakerdict.items():
        LOG.debug('filter in subset {}'.format(label))

        n_words = len(texts)
        # and label.split(';')[1] in spk_with_occurrences
        if n_words > min_words_in_conv and label.split(';')[1] in spk_with_occurrences:
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
    conv_ids = []
    for label, texts in speakers.items():
        labels.append(label.split(";", 1)[1])
        conv_ids.append(label.split(";", 1)[0])
        features.append(texts)

    return np.concatenate(features), np.array(labels), np.array(conv_ids)