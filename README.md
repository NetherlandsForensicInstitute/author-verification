Authorship verification
=======================

Prerequisites
-------------

* A python setup with virtualenv
* the FRIDA dataset

Setup
-----

Clone this repository:

```sh
git clone git@github.com:HolmesNL/author-verification.git
cd author-verification
```

Setup a virtual environment:

```sh
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

To download NLTK files, start `python` and run the following commands:

```py
import nltk
nltk.download('punkt')
```

Copy or extract the FRIDA dataset into the `frida` directory.


Running experiments
-------------------

```sh
classify.py
```


Results
=======

SVM method

n_frequent_words=50*, tokens_per_sample=200*: cllr=0.8203017652820481
n_frequent_words=50*, tokens_per_sample=800: cllr=0.518377257507916
n_frequent_words=50*, tokens_per_sample=1400: cllr=0.3972284625209551
n_frequent_words=150, tokens_per_sample=200*: cllr=0.7941788604945595
n_frequent_words=150, tokens_per_sample=800: cllr=0.3935033064529293
n_frequent_words=150, tokens_per_sample=1400: cllr=0.312044555501085
n_frequent_words=250, tokens_per_sample=200*: cllr=0.781245633087783
n_frequent_words=250, tokens_per_sample=800: cllr=0.37812968584956114
n_frequent_words=250, tokens_per_sample=1400: cllr=0.26279502190033327
