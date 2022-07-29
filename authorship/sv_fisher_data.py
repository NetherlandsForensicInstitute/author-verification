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
        # os.makedirs('.cache', exist_ok=True)
        # path = self._get_cache_path()
        # if os.path.exists(path):
        #     df = pd.read_json(path, orient="index")
        # else:
        #     df = compile_data(self._scores, self._info)
        #     df.to_json(path, orient="index")

        skp_row, skp_col, scores = compile_data(self._scores, self._info)

        return skp_row, skp_col, scores


def compile_data(index_path, info_path):
    '''
    It is in h5 format:
    “enroll”  Names of files
    “scr”     Score enroll vs test in the same order as the above lists. (enroll==scr -> symmetric matrix)
    “test”    Same as above. (for generality since we may have different enroll and test sets sometimes)
    '''

    to_ids = {}
    with open(info_path) as f:
        next(f)

        # for line in f:
        #     # note: it appears that some ids have 4 digits while most of them have 5.
        #     filename, spk_id, trans = line.split('\t', 2)
        #     to_ids[filename.replace('-', '_')] = [spk_id.replace('FISHE', ''),
        #                                           trans.replace('\n', '').replace('/WordWave', '')]

    # load data and keep only the upper triangular matrix (diagonal is also excluded)
    with h5py.File(index_path, "r") as f:
        # get keys
        group_keys = list(f.keys())

        # extract values
        # the elements of enroll and test are numpy bytes so converted to str
        enroll = list(f[group_keys[0]])
        enroll = [a.decode('UTF-8') for a in enroll]
        # scr = pd.DataFrame(f[group_keys[1]][()])  # score as np.array first and then pd.dataframe
        scr = f[group_keys[1]][()]
        test = list(f[group_keys[2]])
        test = [a.decode('UTF-8') for a in test]

    # # [spkid]_[convid+conversation side (a or b)]_[transcriber first letter (B or L)]
    # enroll_full = [to_ids.get(path_str.lower())[0] + '_' +
    #                path_str.replace('fe_03_', '').replace('_A', 'a').replace('_B', 'b') + '_' +
    #                to_ids.get(path_str.lower())[1][0] for path_str in enroll]
    # test_full = [to_ids.get(path_str.lower())[0] + '_' +
    #              path_str.replace('fe_03_', '').replace('_A', 'a').replace('_B', 'b') + '_' +
    #              to_ids.get(path_str.lower())[1][0] for path_str in test]

    # convid+conversation side (a or b)
    enroll_clean = [path_str.replace('fe_03_', '').replace('_A', 'a').replace('_B', 'b') for path_str in enroll]
    test_clean = [path_str.replace('fe_03_', '').replace('_A', 'a').replace('_B', 'b') for path_str in test]

    # spk1 = '51859_11576b_L'
    # spk2 = '98983_10681b_B'
    # enroll_array = np.array(enroll_full)
    # test_array = np.array(test_full)
    # idx_1 = np.where(enroll_array == spk1)
    # idx_2 = np.where(test_array == spk2)
    #
    # prova = scr[idx_1, idx_2].item()

    return np.array(enroll_clean), np.array(test_clean), scr
