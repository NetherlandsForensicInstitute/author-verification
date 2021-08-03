import numpy as np
import pandas as pd
import os
import re


class VocalizeDataSource:
    """
    specific on the vocalize output for frida
    it expects a matrix as a csv file where each speaker/conversation in the row_index (first column) is compared to
    each speaker/conversation in the column_index (first row). It is expected that every speaker appearing in the
    row_index to appear also in the column_index. The value in each position (spk_A, spk_B) is a score returned by
    the speaker verification system VOCALISE. Expected id pattern SP[0-9a]{3,4}-[12]-[1-8]-[1-5], e.g., SP003-1-4-6
    """
    def __init__(self, voc_data, device):
        self._voc_data = voc_data
        self._device = device

    def _get_cache_path(self, device):
        filename_safe = re.sub('[^a-zA-Z0-9_-]', '_', self._voc_data)
        return f'.cache/{filename_safe}_{device}.json'

    def get(self):
        """
        it returns two arrays: the first one holds the ids of the conversations that were compared and the second the
        vocalise output for that comparison
        """
        os.makedirs('.cache', exist_ok=True)
        vocalise_path = self._get_cache_path(self._device)
        if os.path.exists(vocalise_path):
            df = pd.read_json(vocalise_path, orient="index")
        else:
            df = compile_data(self._voc_data, self._device)
            df.to_json(vocalise_path, orient="index")

        voc_pairs = df[['SP_1', 'SP_2']].to_numpy()
        voc_pairs_set = np.apply_along_axis(set, 1, voc_pairs)
        voc_score = df[['value']].to_numpy()

        return voc_pairs_set, voc_score


def compile_data(index_path, device):
    # load data and keep only the upper triangular matrix
    df = pd.read_csv(index_path, index_col=0)
    df.sort_index(inplace=True, axis=0)
    df.sort_index(inplace=True, axis=1)
    df = df.where(np.triu(np.ones(df.shape), k=1).astype(bool))
    df = df.stack(dropna=True).reset_index()
    df.columns = ['SP_1', 'SP_2', 'value']

    # keep only the id of the conversation and throw anything else
    # SP[0-9a]{3,4}-[12]-[1-8]-[1-5] = speaker_id-day-session-device
    conv_id_pattern = "(SP[0-9a]{3,4}-[12]-[1-8]-[1-5])"
    df[['SP_1', 'SP_2']] = df[['SP_1', 'SP_2']].apply(lambda x: x.str.extract(conv_id_pattern, expand=False))

    # in indoor sessions (1 to 4) the telephone is dev 5, in outdoors sessions (so 5 to 8) the telephone is dev 2
    # only the sessions with iphone (so 2, 4, 6, and 8) were transcribed, so we drop the rest
    if device == 'telephone':
        endings = ('2-5', '4-5', '6-2', '8-2')
    elif device == 'headset':
        endings = ('2-1', '4-1', '6-1', '8-1')
    elif device == 'SM58close':
        endings = ('2-2', '4-2')
    elif device == 'AKGC400BL':
        endings = ('2-3', '4-3')
    elif device == 'SM58far':
        endings = ('2-4', '4-4')
    else:
        raise Exception("vocalise_data: no device or incorrect device was given, possible values: telephone, "
                        "headset, SM58close, AKGC400BL, SM58far")

    df = df[df['SP_1'].str.endswith(endings) & df['SP_2'].str.endswith(endings)]

    # to match id pattern as it is in the transcriptions
    df[['SP_1', 'SP_2']] = df[['SP_1', 'SP_2']] \
        .apply(
        lambda x: x.str.extract(conv_id_pattern[:(len(conv_id_pattern) - 7)] + ")", expand=False).str.replace("-", ""))

    return df
