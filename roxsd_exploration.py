import re
from tqdm import tqdm
import json
import collections


def load_data(path):
    with open(path) as f:
        return json.loads(f.read())

prep_clean_data = True
if prep_clean_data:
    all_data = load_data('roxsd/roxsd_after_cleaning.json')
else:
    all_data = load_data('roxsd/roxsd_for_exploration.json')

for_within_par = re.compile("\([a-z ].*\)$")
for_incomplete = re.compile("[a-z].*-$")
for_splits = re.compile("[a-z].*-[a-z].*")
for_other = re.compile("[^a-z\(\)]")

ids = []

num_tokens = []
num_words = []

within_par_to_check = collections.defaultdict(int)
splits_to_check = collections.defaultdict(int)
incomplete_to_check = collections.defaultdict(int)
other_to_check = collections.defaultdict(int)

freq = collections.defaultdict(int)

if __name__ == '__main__':
    for k, v in tqdm(all_data.items()):

        ids.append(k.split("_", 1)[1])

        num_tokens.append(len(v))

        # words within parentheses
        temp_par = list(set(filter(for_within_par.match, v)))
        if len(temp_par) > 0:
            within_par_to_check[k] = temp_par

        # splits words (words joined with '-')
        temp_splits = list(set(filter(for_splits.match, v)))
        if len(temp_splits) > 0:
            splits_to_check[k] = temp_splits

        # incomplete words
        temp_incomplete = list(set(filter(for_incomplete.match, v)))
        if len(temp_incomplete) > 0:
            incomplete_to_check[k] = temp_incomplete

        # checking for missing any special symbol
        other = list(set(filter(for_other.match, v)))
        if len(other) > 0:
            other_to_check[k] = other

        for word in v:
            freq[word] += 1

        clean_list = [i for i in v if not for_within_par.match(i)]
        num_words.append(len(clean_list))

    within_par = [item for sublist in within_par_to_check.values() for item in sublist]
    within_par = list(set(within_par))
    within_par_to_check = {k: ' || '.join(v) for k, v in within_par_to_check.items()}

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

    data = {'num_tokens': num_tokens, 'num_words': num_words, 'within_par': within_par,
            'within_par_to_check': within_par_to_check, 'incomplete': incomplete,
            'incomplete_to_check': incomplete_to_check, 'splits': splits, 'splits_to_check': splits_to_check,
            'other': other, 'other_to_check': other_to_check, 'freq': freq, 'ids_counts': ids_counts}

    if prep_clean_data:
        path = 'roxsd/roxsd_clean_for_streamlit.txt'
    else:
        path = 'roxsd/roxsd_for_streamlit.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=4))

