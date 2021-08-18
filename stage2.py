# analysis pipeline 1.0
# Stage 2 script

# Compatibility Information
#   Written for Python 3.7 on Windows 10

# To execute from within the python interpreter, run:
#	exec(open("stage2.py").read())
# To run from command line:
# 	python stage2.py

# written: 8/17/2021

###################
# GlOBAL SETTINGS #
###################

# true_false_label_ratio=0.5

# single_split_ratio=0.8

# incremental_split_sets=0

###########
# IMPORTS #
###########
import os
import sys
import pickle
import numpy as np
np.random.seed(1337)
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use('Svg')
import matplotlib.pyplot as plt

# train-test split
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


#########################
# STEP 1 - READ IN DATA #
#########################
cwd = os.getcwd()

with open(os.path.join(cwd,'data/processed_data/vector_keywords.pickle'), 'rb') as file:    
	vector_keywords = pickle.load(file)
vocab_size = len(vector_keywords)

proc_reports = pd.read_pickle(os.path.join(cwd,'data/processed_data/proc_reports.df'))

followup_label = list(proc_reports['fu_label'])
d = Counter(followup_label)
print("\nFull Dataset Statistics:")
print("\nNumber of reports\n\tfollow-up\t{} \n\tNon-followup\t{}".format(d[1], d[0]))

#####################################################################
# STEP 2 - Generate 80/20 split set with equal pos and neg examples #
#####################################################################
num_pos_labels = d[1]
num_neg_labels = d[0]

label_count_delta = abs(num_pos_labels-num_neg_labels)

# proc_reports_balanced has half fu_label=1, half fu_label=0
if num_pos_labels > num_neg_labels:
	# more positive than negative, so randomly drop positive label rows (fu_label = 1)
	proc_reports_balanced = proc_reports.drop(proc_reports[proc_reports['fu_label'].eq(1)].sample(label_count_delta).index)
elif num_pos_labels < num_neg_labels:
	# more negative than positive, so randomly drop negative label rows (fu_label = 0)
	proc_reports_balanced = proc_reports.drop(proc_reports[proc_reports['fu_label'].eq(0)].sample(label_count_delta).index)
else:
	# already balance between positive and negative examples
	proc_reports_balanced = proc_reports
	pass

followup_label = list(proc_reports_balanced['fu_label'])

#######################################
# STEP 3 - Make train-test split sets #
#######################################
# 80/20 train/test split using scikitlearn
# Note, only pseudo-random split, because we want to have the same split for testing different models (Repeatable Train-Test Splits)
# https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/
print("\nGenerating 80/20 train-test split sets...")
df_train, df_test = train_test_split(proc_reports_balanced, test_size=0.20, random_state=1)

tr_followup_label = list(df_train['fu_label'])
tr_report_category = list(df_train['report_category'])
d1 = Counter(tr_followup_label)
d2 = Counter(tr_report_category)
print("\nNumber of reports in TRAINING set: \n\tTotal\t\t{} \n\tFU\t\t{} \n\tnon-FU\t\t{} \n\tCode Abdomen\t{} \n\tCode Rec\t{} \n\tOther\t\t{}".format(len(tr_followup_label), d1[1], d1[0], d2[1], d2[2], d2[-2]))

te_followup_label = list(df_test['fu_label'])
te_report_category = list(df_test['report_category'])
d1 = Counter(te_followup_label)
d2 = Counter(te_report_category)
print("\nNumber of reports in TEST set: \n\tTotal\t\t{} \n\tFU\t\t{} \n\tnon-FU\t\t{} \n\tCode Abdomen\t{} \n\tCode Rec\t{} \n\tOther\t\t{}".format(len(te_followup_label), d1[1], d1[0], d2[1], d2[2], d2[-2]))

print("\nActual percent split between train and test sets:\n\tTrain {} \n\tTest {}".format(
	round(len(tr_followup_label)/len(followup_label),5),
	round(len(te_followup_label)/len(followup_label),5)
	))

####################################################
# STEP 4 - Save data for stage2.py (deep learning) #
####################################################
# From these files, the relevant columns for the deep learning model are:
# 'ID','report_clean_tokenized_stemmed_noFU','fu_label'
print("\nSaving train-test split sets...")
df_train.to_pickle(os.path.join(cwd,'data/processed_data/df_train.df'))
df_test.to_pickle(os.path.join(cwd,'data/processed_data/df_test.df'))