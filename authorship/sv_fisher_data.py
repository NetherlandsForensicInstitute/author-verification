import numpy as np
import pandas as pd
import os
import re
import h5py


class SVscoreFisherDataSource:
    """
    todo: UPDATE doc
    """

    def __init__(self, scores, info):
        self._scores = scores
        self._info = info

    def _get_cache_path(self):
        filename_safe = re.sub('[^a-zA-Z0-9_-]', '_', self._scores)
        return f'.cache/{filename_safe}.json'

    def get(self):
        """
        it returns two arrays: the first one holds the ids of the conversations that were compared and the second the
        acoustic score for that comparison
        """
        os.makedirs('.cache', exist_ok=True)
        path = self._get_cache_path()
        if os.path.exists(path):
            df = pd.read_json(path, orient="index")
        else:
            df = compile_data(self._scores, self._info)
            df.to_json(path, orient="index")

        # pairs = df[['SP_1', 'SP_2']].to_numpy()
        # pairs = np.apply_along_axis(lambda a: str(a[0] + a[1]), 1, pairs)
        # scores = df[['value']].to_numpy()

        pair_score = dict({})
        for A, B in zip(df['SP_1'], df['SP_2']):
            pair_score[str(A + '.' + B)] = df['value']

        return pair_score


def compile_data(index_path, info_path):
    '''
    It is in h5 format:
    “enroll”  Names of files
    “scr”     Score enroll vs test in the same order as the above lists. (enroll==scr -> symmetric matrix)
    “test”    Same as above. For generality since we may have different enroll and test sets sometimes
    '''

    to_ids = {}
    with open(info_path) as f:
        next(f)

        for line in f:
            # note: it appears that some ids have 4 digits while most of them have 5.
            filename, spk_id, trans = line.split('\t', 2)
            to_ids[filename.replace('-', '_')] = [spk_id.replace('FISHE', ''),
                                                  trans.replace('\n', '').replace('/WordWave', '')]

    # load data and keep only the upper triangular matrix (diagonal is also excluded)
    with h5py.File(index_path, "r") as f:
        # get keys
        group_keys = list(f.keys())

        # extract values
        # the elements of enroll and test are numpy bytes so converted to str
        enroll = list(f[group_keys[0]])
        enroll = [a.decode('UTF-8') for a in enroll]
        scr = pd.DataFrame(f[group_keys[1]][()])  # score as np.array first and then pd.dataframe
        test = list(f[group_keys[2]])
        test = [a.decode('UTF-8') for a in test]

    # [spkid]_[convid+conversation side (a or b)]_[transcriber first letter (B or L)]
    enroll_full = [to_ids.get(path_str.lower())[0] + '_' +
                   path_str.replace('fe_03_', '').replace('_A', 'a').replace('_B', 'b') + '_' +
                   to_ids.get(path_str.lower())[1][0] for path_str in enroll]
    test_full = [to_ids.get(path_str.lower())[0] + '_' +
                 path_str.replace('fe_03_', '').replace('_A', 'a').replace('_B', 'b') + '_' +
                 to_ids.get(path_str.lower())[1][0] for path_str in enroll]

    prova = []
    for val in to_ids.values():
        prova.append(len(val[0]))

    scr.columns = test_full
    scr.set_index(np.array(enroll_full), inplace=True)
    scr.sort_index(inplace=True, axis=0)
    scr.sort_index(inplace=True, axis=1)
    scr = scr.where(np.triu(np.ones(scr.shape), k=1).astype(bool))
    scr = scr.stack(dropna=True).reset_index()
    scr.columns = ['SP_1', 'SP_2', 'value']

    return scr
