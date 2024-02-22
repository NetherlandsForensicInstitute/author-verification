import collections
import json
import logging
import os
import re
import jiwer
import glob

import numpy as np

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# from boilerplate import fileio


LOG = logging.getLogger(__name__)


class DataSource:
    def __init__(self, data_path, data_name, ground_truth, extra_info=None, remove_filler_sounds=False,
                 expand_contractions=False, n_frequent_words=200, min_num_of_words=1):

        self._data_path = data_path
        self._data_name = data_name
        self._ground_truth = ground_truth
        self._extra_info = extra_info
        self._remove_filler_sounds = remove_filler_sounds
        self._expand_contractions = expand_contractions
        self._n_freqwords = n_frequent_words
        self._min_num_of_words = min_num_of_words

    def _get_cache_path(self):
        path_safe = re.sub('[^a-zA-Z0-9_-]', '_', os.path.basename(self._data_path))

        if self._data_name in path_safe:
            base_name = path_safe
        else:
            base_name = self._data_name + '_' + path_safe

        filename = (base_name + '_gt_' + str(self._ground_truth)[0] +
                    '_rm_fillers_' + str(self._remove_filler_sounds)[0] +
                    '_exp_cont_' + str(self._expand_contractions)[0])
        return f'.cache/{filename}.json'

    def get(self):
        os.makedirs('.cache', exist_ok=True)
        speakers_path = self._get_cache_path()
        if os.path.exists(speakers_path):
            LOG.debug(f'using cache file: {speakers_path}')
            speakers_wordlist = load_data(speakers_path)
        else:
            speakers_wordlist = compile_data(self._data_name, self._data_path, self._ground_truth, self._extra_info,
                                             self._remove_filler_sounds, self._expand_contractions)
            store_data(speakers_path, speakers_wordlist)

        # extract a list of frequent words
        wordlist = [word for word, freq in get_frequent_words(speakers_wordlist, self._n_freqwords)]

        # build a dictionary of feature vectors
        speakers_conv = filter_texts(speakers_wordlist, wordlist, self._min_num_of_words)

        # convert to X, y (= spk id) but keep the conversation id
        X, y, conv_ids = to_vector_size(speakers_conv, self._data_name)

        return X, y, conv_ids

    def __repr__(self):
        return f'data(freqwords={self._n_freqwords})'


class CreateFeatureVector:
    def __init__(self, wordlist):
        self.wordlist = wordlist

    def __call__(self, text):
        """
        it takes a list of words used in a conversation by one speaker and returns the counts of the words in the
        wordlist
            :param text: a list of words
        """

        if len(text) == 0:
            return []
        else:
            vectorizer = TfidfVectorizer(analyzer='word', use_idf=False, norm=None, vocabulary=self.wordlist,
                                         tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
            return vectorizer.fit_transform([text]).toarray()


def load_data(path):
    with open(path) as f:
        return json.loads(f.read())


def store_data(path, speakers):
    with open(path, 'w') as f:
        f.write(json.dumps(speakers))


def read_session(lines, data_name, ground_truth, remove_filler_words=False, expand_contractions=False):
    """
    it takes a path to a transcription file and returns a dictionary that maps conversation id to a list of words.
        :param lines: <class '_io.TextIOWrapper'>
        :param data_name: name/id of the dataset (e.g., frida or fisher)
        :param ground_truth: whether the input are manual transcriptions (then True) or ASR output (then False)
        :param remove_filler_words: whether filler sounds such as uh, uh huh, ehm to be removed or not
        :param expand_contractions:
    """
    lines_to_words = lines.read()
    words_to_remove = []

    if ground_truth:
        if data_name == 'frida':
            # for FRIDA remember:
            # *v: non-Dutch words,  *n: new non-existing words, *s: street  words,
            # *a: incomplete words, *u: distorted words, *x: unclear word,
            # xxx: unclear utterances, vvv: non-Dutch sentences, ggg: sounds made by the speaker

            lines_to_words = re.sub('[0-9]*\.[0-9]*\t', '', lines_to_words)  # to remove timestamps
            lines_to_words = re.sub('[A-Za-z]*\*[anuxANUX]{1}', '',
                                    lines_to_words)  # to remove words with *n, *a, *u, and *x
            lines_to_words = re.sub('[A-Za-z]*\*[etV]{1}', '', lines_to_words)  # unknown notation
            lines_to_words = re.sub('[A-Za-z]*\*op', '', lines_to_words)  # a mistake?

            lines_to_words = (lines_to_words.replace('start\tend\ttext\n', '').replace('-', ' ')
                              .replace('\n', ' ').replace('*v', '').replace('*s', '')
                              .replace('*n', ''))

            lines_to_words = re.sub('[A-Za-z]*\*', '', lines_to_words)  # for words with missing notation

            if expand_contractions:
                lines_to_words = (lines_to_words.replace('\'t', 'het').replace('z\'n', 'zijn')
                                  .replace('\zo\'n', 'zo een').replace('\'m', 'hem')
                                  .replace('\'k', 'ik').replace('da\'s', 'dat is')
                                  .replace('\'s', ' is').replace('d\'r', 'haar')
                                  .replace(' \'n', ' een'))

            words_to_remove = ['xxx', 'ggg', 'gggggg', 'vvv']

            if remove_filler_words:
                words_to_remove.extend(['a', 'e', 'oo', 'oe', 'aa', 'ee', 'mm', 'he', 'eh', 'uh', 'hm', 'jo',
                                        'uhu', 'uhm', 'eeh', 'nah', 'joo', 'ach', 'hoo', 'woo'])

        elif data_name == 'fisher':
            lines_to_words = re.sub('[0-9]*\.[0-9]*\ [0-9]*\.[0-9]*\ [AB]:', '', lines_to_words)
            lines_to_words = re.sub('\[\[[a-zA-Z]*\]\]', '', lines_to_words)  # -> <SKIP>
            lines_to_words = re.sub('\[[a-zA-Z]*\]', '', lines_to_words)  # -> <SOUND>
            lines_to_words = re.sub('\(\(', '', lines_to_words)  # remove (( and )) but keep text inside
            lines_to_words = re.sub('\)\)', '', lines_to_words)
            lines_to_words = re.sub('[a-zA-Z]*- ', ' ', lines_to_words)
            lines_to_words = lines_to_words.replace('\n', ' ')
            lines_to_words = lines_to_words.replace('-', ' ')  # so uh-huh is uh huh, in this point because
            # words like a three-year-old and
            # son-in-law are transcribed with -
            # between the words

            if remove_filler_words:
                words_to_remove.extend(['er', 'eh', 'ah', 'hm', 'ha', 'um', 'em', 'uh', 'oh', 'mm', 'ya', 'hmm', 'ohh',
                                        'huh', 'gee', 'mhm', 'nah', 'ahem', 'jeeze'])

            if expand_contractions:
                lines_to_words = jiwer.ExpandCommonEnglishContractions()(lines_to_words)

    elif not ground_truth:
        if data_name == 'frida':
            lines_to_words = (lines_to_words.replace('Eén', 'één').replace('ok', 'oké').replace('oke', 'oké')
                              .replace('haha', '').replace('hahaha', ''))

            if expand_contractions:
                # only treat the ones I saw
                lines_to_words = (lines_to_words.replace('z\'n', 'zijn').replace('\zo\'n', 'zo een')
                                  .replace('m\'r', 'maar'))

            if remove_filler_words:
                words_to_remove.extend(['a', 'e', 'o', 'z', 'he', 'hé', 'hè', 'uh', 'oh', 'ah', 'eh', 'ha', 'hm', 'jo',
                                        'yo', 'ach', 'huh', 'hmm', 'ehh', 'ehm', 'uhm', 'eee', 'joh', 'oei', 'oeh',
                                        'mmhmm'])

        elif data_name == 'fisher':
            lines_to_words = lines_to_words.replace('ok', 'okay')

            if remove_filler_words:
                words_to_remove.extend(['oh', 'um', 'uh', 'mm', 'ah', 'ha', 'huh', 'hmm', 'uhhuh', 'mmhmm'])

            if expand_contractions:
                lines_to_words = jiwer.ExpandCommonEnglishContractions()(lines_to_words)

    general_transformations = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.RemoveSpecificWords(words_to_remove),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ReduceToListOfListOfWords()
    ])

    lines_to_words = general_transformations(lines_to_words)
    return lines_to_words[0]


def compile_data(data_name, index_path, ground_truth, info_path=None, remove_filler_words=False,
                 expand_contractions=False):
    """
    it takes a path to folder. It loops through all the .txt files in the folder and returns a dictionary
    where the key is the id of a conversation and the value a list of texts
        :param data_name:
        :param index_path: str
        :param ground_truth:
        :param info_path:
        :param remove_filler_words:
        :param expand_contractions:
    """
    to_ids = {}
    if data_name == 'fisher' and info_path is not None:
        with open(info_path) as f:
            next(f)
            for line in f:
                # each line has the filename, id of speaker, transcriber (LDC or BBN), sex of the speaker,
                # whether the speaker is native english or not, and how many conversations the speaker had
                # (see update_info_file.py or folder fisher_metadata)
                filename_full, spk_id, trans, sx, dl, count = line.split('\t', 5)
                filename = (filename_full.replace('fe_03_', '').replace('.txt', '').replace('_a', 'a')
                            .replace('-a', 'a').replace('_b', 'b').replace('-b', 'b'))
                to_ids[filename] = [spk_id, trans, sx, count.replace('\n', '')]

    conversations = collections.defaultdict(list)  # create empty dictionary list
    files = glob.glob(os.path.join(index_path, "*.txt"))
    file_to_check = None

    for filepath in tqdm(files):
        path_str = os.path.basename(filepath)

        if data_name == 'frida':
            # conversation id from SPXXX?-S-D-N.txt (for ref) or SPXXX?-S-D-N_raw.txt (for asr) to SPXXXSD
            # where '?' is most of the times '' but sometimes is 'a'
            if ground_truth:
                conv_id = path_str[:len(path_str) - 6].replace('-', '')
            else:
                conv_id = path_str[:len(path_str) - 10].replace('-', '')

        elif data_name == 'fisher':
            # from fe_03_XXXXX{-_}{ab}.txt to XXXXX{ab} (for ref it is '-' and for asr is '_')
            file_to_check = (path_str.replace('fe_03_', '').replace('.txt', '')
                             .replace('_a', 'a').replace('-a', 'a')
                             .replace('_b', 'b').replace('-b', 'b'))

            conv_id = to_ids.get(file_to_check)[0] + '_' + file_to_check
        else:
            conv_id = 'error'
            print('unexpected data name')
            break

        with open(filepath, "r") as f:

            if data_name == 'fisher' and (to_ids.get(file_to_check)[2] == '' or to_ids.get(file_to_check)[3] == '1'):
                continue
            else:
                text = read_session(f, data_name, ground_truth, remove_filler_words, expand_contractions)
                if len(text) > 0:  # to exclude empty files
                    conversations[conv_id] = text
                # if 'ewa' in text:
                #     print(conv_id)
                # else:
                #     print('empty file = ' + conv_id)

    return conversations


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
            freq[word] += 1
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return freq[:n]


def filter_texts(conversations, wordlist, min_num_of_words):
    """
    it returns a dict that each key correspond to a conversation of one speaker and its value is a vector of length
    equal to the length of the wordlist and its values are the relative frequencies of the words in the wordlist for
    that speaker/conversation.

    :param conversations: dict of all words used per conversation
    :param wordlist: the n most freq words in corpus
    :param min_num_of_words:
    """
    f = CreateFeatureVector(wordlist)

    filtered = {}
    for label, text in conversations.items():
        LOG.debug('filter in subset {}'.format(label))
        ltext = len(text)
        filtered_text = list(f(text))
        if ltext > min_num_of_words:
            filtered[label] = [i / ltext for i in filtered_text]

    return filtered


def to_vector_size(speakers, data_name):
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
        if data_name == 'frida':
            speaker_ids.append(conv_id[2:(len(conv_id) - 2)])
        elif data_name == 'fisher':
            speaker_ids.append(conv_id.split('_')[0])
        conv_ids.append(conv_id)

    return np.concatenate(features), np.array(speaker_ids), np.array(conv_ids)
