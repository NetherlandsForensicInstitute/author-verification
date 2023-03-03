import numpy as np
import pandas as pd
import os
import re
import h5py


class SVscoreFisherDataSource:
    """
    todo: UPDATE doc
    """

    def __init__(self, scores_file):
        self._scores_file = scores_file

    def get(self):
        """
        it returns 3 arrays: the first and second arrays are the ids of the conversations and the third, which is a
        matrix, the acoustic scores of the comparison.
        """
        skp_row, skp_col, scores = compile_data(self._scores_file)

        return skp_row, skp_col, scores


def compile_data(index_path):
    '''
    It is in h5 format:
    “enroll”  Names of files
    “scr”     Score enroll vs test in the same order as the above lists. (enroll==scr -> symmetric matrix)
    “test”    Same as above. (for generality since we may have different enroll and test sets sometimes)
    '''

    # load data
    with h5py.File(index_path, "r") as f:
        # get keys
        group_keys = list(f.keys())

        # extract values
        # the elements of enroll and test are numpy bytes so converted to str
        enroll = list(f[group_keys[0]])
        enroll = [a.decode('UTF-8') for a in enroll]  # score as np.array
        scr = f[group_keys[1]][()]
        test = list(f[group_keys[2]])
        test = [a.decode('UTF-8') for a in test]

    # conv_id+conversation side (a or b)
    enroll_clean = [path_str.replace('fe_03_', '').replace('_A', 'a').replace('_B', 'b') for path_str in enroll]
    test_clean = [path_str.replace('fe_03_', '').replace('_A', 'a').replace('_B', 'b') for path_str in test]

    return np.array(enroll_clean), np.array(test_clean), scr
