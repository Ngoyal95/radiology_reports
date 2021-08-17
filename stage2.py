# analysis pipeline 1.0
# Stage 2 script

# Compatibility Information
#   Written for Python 3.7 on Windows 10

# To execute from within the python interpreter, run:
#	exec(open("stage2.py").read())
# To run from command line:
# 	python stage2.py

# written: 7/28/2021

###################
# GlOBAL SETTINGS #
###################

###########
# IMPORTS #
###########

import os
import sys
import pickle
import numpy as np
np.random.seed(1337)
import pandas as pd

import matplotlib
matplotlib.use('Svg')
import matplotlib.pyplot as plt


################
# READ IN DATA #
################
cwd = os.getcwd()

with open(os.path.join(cwd,'data/processed_data/vector_keywords.pickle'), 'rb') as file:    
	vector_keywords = pickle.load(file)
vocab_size=len(vector_keywords)

df_reports = pd.read_pickle(os.path.join(cwd,'data/processed_data/proc_reports.df'))




###########################################################
# STEP 5 - Determine max report len, and then pad reports #
###########################################################
reports_train = [rep for rep in df_train['report_clean_tokenized_stemmed_noFU']]
reports_test = [rep for rep in df_test['report_clean_tokenized_stemmed_noFU']]

report_lengths = [len(x) for x in reports_train]

print("\nReport lengths (pure semantic content, tokenized):\n\tAverage:\t{} \n\tMax:\t{} \n\tMin:\t{}\n".format(sum(report_lengths)/len(report_lengths), max(report_lengths), min(report_lengths)))
print("\n\tLongest report (idx {}):\n".format(report_lengths.index(max(report_lengths))))
# a=np.histogram(report_lengths)
# plt.hist(a)
# plt.savefig('pre_pad-trunc_report_lengths.png')

# convert each report to a sequence of integers
tokenizer = Tokenizer(num_words=vocab_size, lower=True, char_level=False)
tokenizer.fit_on_texts(reports_train + reports_test)

# https://stackoverflow.com/questions/51956000/what-does-keras-tokenizer-method-exactly-do
word_seq_train = tokenizer.texts_to_sequences(reports_train)
word_seq_test = tokenizer.texts_to_sequences(reports_test)
word_index = tokenizer.word_index

# Pad word_seq_train
x_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len, padding='post', dtype=object, truncating='post')
x_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len, padding='post', dtype=object, truncating='post')

y_train = np.asarray([x for x in df_train['fu_label']]).astype(np.int)
y_test = np.asarray([x for x in df_test['fu_label']]).astype(np.int)

x_train = x_train.astype('int')
y_train = y_train.astype('int')

x_test = x_test.astype('int')
y_test = y_test.astype('int')

