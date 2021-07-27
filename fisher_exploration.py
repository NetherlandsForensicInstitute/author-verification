
import re
from tqdm import tqdm
import json
import collections


def load_data(path):
    with open(path) as f:
        return json.loads(f.read())


all_data = load_data('fisher/fisher_for_exploration.json')

for_incomplete = re.compile("[a-z].*-$")
for_splits = re.compile("[a-z].*-[a-z].*")
for_other = re.compile("[^a-z<>\-]")

transcriber = []
ids = []
ldc_ids = []
bbn_ids = []

num_tokens = []
no_tokens = []
num_words = []
no_words = []

num_unk = []
num_guess = []
num_sounds = []
num_skips = []

num_incomplete = []
incomplete_to_check = collections.defaultdict(int)
num_splits = []
splits_to_check = collections.defaultdict(int)
miss_to_check = []
other_to_check = collections.defaultdict(int)
freq = collections.defaultdict(int)
freq_ldc = collections.defaultdict(int)
freq_bbn = collections.defaultdict(int)
freq_bi = collections.defaultdict(int)

if __name__ == '__main__':

    for k, v in tqdm(all_data.items()):

        ids.append(k.split('_')[0])
        # keep transcriber to check the differences between them
        if bool(re.search('BBN', k)):
            transcriber.append('bbn')
            bbn_ids.append(k.split('_')[0])
        else:
            transcriber.append('ldc')
            ldc_ids.append(k.split('_')[0])

        num_tokens.append(len(v))  # num of tokens (everything is included)
        if len(v) < 20:
            no_tokens.append(k)

        num_unk.append(v.count('<UNK>'))  # count of unclear speech
        num_guess.append(v.count('<GUESS>'))  # count of uncertain transcription
        num_sounds.append(v.count('<SOUND>'))  # count of sounds
        num_skips.append(v.count('<SKIP>'))  # count of skip

        # incomplete words
        temp_incomplete = list(filter(for_incomplete.match, v))
        num_incomplete.append(len(temp_incomplete))
        if len(temp_incomplete) > 0:
            temp_incomplete = list(set(temp_incomplete))
            incomplete_to_check[k] = temp_incomplete

        # splits words (words joined with '-')
        temp_splits = list(filter(for_splits.match, v))
        num_splits.append(len(temp_splits))
        if len(temp_splits) > 0:
            temp_splits = list(set(temp_splits))
            splits_to_check[k] = temp_splits

        # number of words excluding incomplete, unclear, and sounds
        clean_list = [i for i in v if not for_incomplete.match(i) and i != '<UNK>' and i != '<GUESS>' and i != '<SOUND>' and i != '<SKIP>']
        num_words.append(len(clean_list))

        if len(clean_list) < 20:
            no_words.append(k)

        # did i miss anything?
        if v.count('((') > 0 or v.count('))') > 0 or v.count('[[') > 0 or v.count(']]') > 0:
            miss_to_check.append(k)

        other = list(filter(for_other.match, v))
        if len(other) > 0:
            other = list(set(other))
            other_to_check[k] = other

        # get frequent words
        if len(clean_list) > 1:
            for word in clean_list:
                freq[word] += 1
                if bool(re.search('BBN', k)):
                    freq_bbn[word] += 1
                else:
                    freq_ldc[word] += 1

            for i in range(len(v)-1):
                if v[i] in clean_list and v[i+1] in clean_list:
                    bigram = v[i] + ' ' + v[i+1]
                    freq_bi[bigram] += 1
                else:
                    continue

    incomplete = [item for sublist in incomplete_to_check.values() for item in sublist]
    incomplete = list(set(incomplete))
    incomplete_to_check = {k: ' || '.join(v) for k, v in incomplete_to_check.items()}

    splits = [item for sublist in splits_to_check.values() for item in sublist]
    splits = list(set(splits))
    splits_to_check = {k: ' || '.join(v) for k, v in splits_to_check.items()}

    other = [item for sublist in other_to_check.values() for item in sublist]
    other = list(set(other))
    other_to_check = {k: ' || '.join(v) for k, v in other_to_check.items()}

    ids_occ = [ids.count(v) for v in list(set(ids))]
    ids_counts = [ids_occ.count(v) for v in range(1, max(ids_occ)+1)]

    ldc_ids_occ = [ldc_ids.count(v) for v in list(set(ldc_ids))]
    ldc_ids_counts = [ldc_ids_occ.count(v) for v in range(1, max(ids_occ)+1)]

    bbn_ids_occ = [ids.count(v) for v in list(set(bbn_ids))]
    bbn_ids_counts = [bbn_ids_occ.count(v) for v in range(1, max(ids_occ)+1)]

    data = {'transcriber': transcriber, 'num_unk': num_unk, 'num_guess': num_guess, 'num_sounds': num_sounds,
            'num_skips': num_skips, 'num_tokens': num_tokens, 'no_tokens': no_tokens, 'num_words': num_words,
            'no_words': no_words, 'incomplete': incomplete, 'incomplete_to_check': incomplete_to_check,
            'num_incomplete': num_incomplete, 'splits': splits, 'splits_to_check': splits_to_check,
            'num_splits': num_splits, 'miss_to_check': miss_to_check, 'other': other, 'other_to_check': other_to_check,
            'freq': freq, 'freq_ldc': freq_ldc, 'freq_bbn': freq_bbn, 'freq_bi': freq_bi, 'ids_counts': ids_counts,
            'ldc_ids_counts': ldc_ids_counts, 'bbn_ids_counts': bbn_ids_counts}
    path = 'fisher/data_for_streamlit.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=4))
