import collections
import json
import logging
import os
import re
import string
import numpy as np

from nltk.tokenize import WhitespaceTokenizer
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from boilerplate import fileio


LOG = logging.getLogger(__name__)