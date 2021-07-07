import numpy as np
import pandas as pd


class VocalizeDataSource:
    # specific on the vocalize output for frida
    def __init__(self, voc_data, device='telephone'):
        self._voc_data = voc_data
        self._device = device

    def get(self):
        # load data and keep only the upper triangular matrix
        df = pd.read_csv(self._voc_data, index_col=0)
        df.sort_index(inplace=True, axis=0)
        df.sort_index(inplace=True, axis=1)
        df = df.where(np.triu(np.ones(df.shape), k=1).astype(bool))
        df = df.stack(dropna=True).reset_index()
        df.columns = ['SP_1', 'SP_2', 'value']

        # keep only the id of the conversation
        # SP[0-9a]{3,4}-[12]-[1-8]-[1-5] = speaker_id-day-session-device
        conv_pattern = "(SP[0-9a]{3,4}-[12]-[1-8]-[1-5])"
        df[['SP_1', 'SP_2']] = df[['SP_1', 'SP_2']].apply(lambda x: x.str.extract(conv_pattern, expand=False))

        # be careful (in 5 to 8 sessions the telephone is dev 2)!!
        # only the sessions with nokia (so 2, 4, 6, and 8) were transcribed, so we drop the rest
        if self._device == 'telephone':
            endings = ('2-5', '4-5', '6-2', '8-2')
        elif self._device == 'headset':
            endings = ('2-1', '4-1', '6-1', '8-1')
        elif self._device == 'SM58close':
            endings = ('2-2', '4-2')
        elif self._device == 'AKGC400BL':
            endings = ('2-3', '4-3')
        elif self._device == 'SM58far':
            endings = ('2-4', '4-4')
        else:
            endings = ''
            print("vocalise: no device or incorrect device was given, no filter was applied")

        df = df[df['SP_1'].str.endswith(endings) & df['SP_2'].str.endswith(endings)]

        df[['SP_1', 'SP_2']] = df[['SP_1', 'SP_2']] \
            .apply(lambda x: x.str.extract(conv_pattern[:(len(conv_pattern) - 7)] + ")", expand=False).str.replace("-", ""))

        voc_pairs = df[['SP_1', 'SP_2']].to_numpy()
        voc_pairs_set = np.apply_along_axis(set, 1, voc_pairs)
        voc_score = df[['value']].to_numpy()

        return (voc_pairs_set, np.array(voc_score))
