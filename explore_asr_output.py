import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import re
import jiwer

# with open('fisher/fisher_selected_final_ref.txt') as f:
#     ground_truth = f.readline()
#     ground_truth = re.sub('\(\(', '', ground_truth)
#     ground_truth = re.sub('\)\)', '', ground_truth)
#
# with open('fisher/fisher_selected_final_hyp.txt') as f:
#     hypothesis = f.readline()

with open('fisher/fisher_trans/fe_03_00001_a.txt') as f:
    ground_truth = f.read()
    starttime = re.findall('[0-9]*\.[0-9]*', ground_truth)[0]
    endtime = re.findall('[0-9]*\.[0-9]*', ground_truth)[-1]

    ground_truth = re.sub('[0-9]*\.[0-9]*\ [0-9]*\.[0-9]*\ [AB]:', '', ground_truth)
    ground_truth = re.sub('\(\(', '', ground_truth)
    ground_truth = re.sub('\)\)', '', ground_truth)
    ground_truth = re.sub('\[\[[a-z]*\]\]', '', ground_truth)
    ground_truth = re.sub('\[[a-z]*\]', '', ground_truth)
    ground_truth = ground_truth.replace('\n', ' ')


with open('fisher/asr_output_with_times/FE_03_00001_A.txt') as f:
    hypothesis = ""
    for line in f:
        startsat = re.findall('\d{1,}\.\d{1,}', line)[0]
        endsat = re.findall('\d{1,}\.\d{1,}', line)[-1]
        if (float(startsat) >= float(starttime)) & (float(endsat) <= float(endtime)):
            temp = re.sub('[0-9]*\.[0-9]*\t[0-9]*\.[0-9]*\t', '', line)
            temp = temp.replace('\n', ' ')
            hypothesis = hypothesis + ' ' + temp
        else:
            continue



# Word Error Rate (WER), Match Error Rate (MER), Word Information Lost (WIL) and Word Information Preserved (WIP)
# ground_truth = ["hello world", "i like monthy python"]
# hypothesis = ["hello duck", "i like python"]
#
# error = jiwer.wer(ground_truth, hypothesis)

measures = jiwer.compute_measures(ground_truth, hypothesis)

# wer = measures['wer']
# mer = measures['mer']
# wil = measures['wil']
# wip = measures['wip']

# ground_truth_2 = ["i can spell", "i hope"]
# hypothesis_2 = ["i kan cpell", "i hop"]
#
# cer = jiwer.cer(ground_truth_2, hypothesis_2)
#
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
])

