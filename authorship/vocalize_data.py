import numpy as np
import pandas as pd


class VocalizeDataSource:
    # specific on the vocalize output for frida
    def __init__(self, voc_data, conversation_ids):
        self._conversation_ids = conversation_ids
        self._voc_data = voc_data

    def get(self):
        # load data and keep only the upper triangular matrix
        df = pd.read_csv(self._voc_data, index_col=0)
        df.sort_index(inplace=True, axis=0)
        df.sort_index(inplace=True, axis=1)
        df = df.where(np.triu(np.ones(df.shape), k=1).astype(bool))
        df = df.stack(dropna=True).reset_index()
        df.columns = ['SP_1', 'SP_2', 'Value']

        # keep only the id of the conversation
        conv_pattern = "(SP[0-9a]{3,4}-[12]-[1-8]-[1-5])"
        df[['SP_1', 'SP_2']] = df[['SP_1', 'SP_2']].apply(lambda x: x.str.extract(conv_pattern, expand=False))

        # be careful (in 5 tp 9 sessions the telephone is dev 3)!! + i want to have as option to choose recording device
        df = df[df['SP_1'].str.endswith('5') & df['SP_2'].str.endswith('5')]

        df[['SP_1', 'SP_2']] = df[['SP_1', 'SP_2']] \
            .apply(
            lambda x: x.str.extract(conv_pattern[:(len(conv_pattern) - 7)] + ")", expand=False).str.replace("-", ""))

        # to check it with different voc output
        df = df[df['SP_1'].isin(self._conversation_ids) & df['SP_2'].isin(self._conversation_ids)]

        return df
