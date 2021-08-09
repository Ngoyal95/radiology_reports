# analysis pipeline 1.0
# Stage 1 script

# Compatibility Information
# 	Written for Python 3.7 on Windows 10

# To execute from within the python interpreter, run:
#	exec(open("stage1.py").read())
# To run from command line:
# 	python stage1.py

###################
# GlOBAL SETTINGS #
###################
use_glove = 0
use_biobert = 0

###########
# IMPORTS #
###########
import os
import re
import string
import sys
import pickle

from collections import Counter

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import gensim
from gensim.models import FastText, KeyedVectors

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import numpy as np
from numpy import zeros

# imports to utilize BioBERT embeddings
sys.path.append('/biobert_embedding/biobert_embedding/')
from biobert_embedding.embedding import BiobertEmbedding

# PCA on word embeddings
from sklearn.decomposition import PCA

# train-test split
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

########################
# FUNCTION DEFINITIONS #
########################
# https://stackoverflow.com/questions/57030670/how-to-remove-punctuation-and-numbers-during-tweettokenizer-step-in-nlp
def clean_text(text):
	# remove numbers
	text_nonum = re.sub(r'\d+', '', text)
	# remove punctuations and convert characters to lower case
	text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation]) 
	# substitute multiple whitespace with single whitespace
	# Also, removes leading and trailing whitespaces
	text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
	return text_no_doublespace


#########################
# STEP 1 - READ IN DATA #
# functional 7/17/21	#
#########################
# import reports
cwd = os.getcwd()
input_file = os.path.join(cwd,'data/goyal_reports_with_codes.csv')

# read in data to pandas dataframe
raw_data = pd.read_csv(input_file)

# drop the existing id column that is taken from CSV file, replace with a new index col
raw_data.drop(['ID'], axis=1, inplace=True)
raw_data['idx'] = raw_data.index

###############################################################
# STEP 2 - Categorize reports (code abdomen, code rec, other) #
###############################################################
# report_category values are {1: code abdomen, 2:code rec, -1: code cancer, -2: other}
report_category = []
raw_data_copy = raw_data

for ind in raw_data_copy.index:
	report = raw_data_copy['report'][ind]
	if "FOCAL_MASS_SUMMARY" in report:
		# Code Abdomen
		report_category.append(1)
	elif "NON-EMERGENT ACTIONABLE FINDINGS" in report:
		# Code Rec
		report_category.append(2)
	elif "START INTERVAL ONCOLOGIC RESPONSE ASSESSMENT (ABDOMEN-PELVIS)" in report:
		# Code Cancer
		# Drop these reports because they may skew our model since all these patients have known malignancy
		report_category.append(-1)
	else:
		# case where F/U not present
		report_category.append(-2)

# make new column with the report_category as new column
raw_data_copy['report_category'] = report_category

# Basic statistics
d = Counter(report_category)
print("\n\n------BEGIN STAGE1 SCRIPT PRINT OUTPUT------\n")
print("Number of\n\tCode Abdomen\t{} \n\tCode Rec\t{} \n\tCode Cancer\t{} \n\tNoneOfAbove\t{} \n\t ".format(
	d[1], d[2], d[-1], d[-2]
	))

# Drop all Code Cancer reports (flagged 'report_category' == -1)
raw_data_copy.drop(raw_data_copy[raw_data_copy['report_category'] == -1].index, inplace=True)

###############################################################
# STEP 3 - Obtain fundamental semantic content of each report #
# note, functional as of 7/17/21
###############################################################
# Note 7/20/2021 - fastText has a tokenizer, possible option to use that instead of nltk
# Obtaining semantic content
# (Per Lou 2020, JDI) lower case, remove punctuation/symbols/numbers, tokenize, remove stop words, porter stem (PRESENT CODE USES THIS)

# Codes to be used for Follow-up label
code_abd_fu_codes = ['C3','C4','C5']
code_rec_fu_codes = ['REC2b','REC3','REC3a','REC3b','REC3c','REC4','REC5']

# use this for sanity check to make sure we capture all the codes correctly
# code_abd_fu_codes = ['C1','C2','C3','C4','C5','C6','C7','C99']
# code_rec_fu_codes = ['REC0','REC1','REC2a','REC2b','REC3','REC3a','REC3b','REC3c','REC4','REC5','REC99']

# Code Abdomen
code_abd_regex = re.compile("(?sim)FOCAL_MASS_SUMMARY.*?\}(?!,)")	# regex uses the negative lookahead \}(?!,) to find closing brace not followed by a commma
# regex to capture specific Code Abdomen option to determine if true F/U present, call with re.findall to get all options listed
code_abd_regex_coding = re.compile("(?sim)(\{C(1|2|3|4|5|6|7|99|X):)")

# Code Rec
code_rec_regex = re.compile("(?ims)(NON-EMERGENT ACTIONABLE FINDINGS)[\w\n\s\d,.!?\\\/-]*(Recommendation:[\w\s\d,.!?\\\/\-();:\[\]\{\}]*(\[REC[\d]+[\w]?\])[\n\s\d,.!?\\\/-]*\n*)")
# Regex to capture the REC code associated with the Code Rec text, call with re.findall to get all options listed
code_rec_regex_coding = re.compile("(?sim)(\[REC(0|1|2a|2b|3a|3b|3c|4|5|99)\])")

# Act 112 Notification
# Detect and redact using regex to capture:
# 	FOLLOW-UP NOTICE: ......... [FOL3M]
act_112_regex = re.compile("(?ims)(FOLLOW-UP NOTICE:[\n\w\s\d,.!?\\\/\-();:\[\]\{\}]*\[FOL3M\])")

# Store non-tokenized copy of report text
reports_clean = []
reports_clean_noFU = []

# Store tokenized copy of report text
reports_clean_tokenized = []
reports_clean_tokenized_noFU = []

# For Code Abdomen/Rec reports, use this list to store their CODE text content
followup_text = []
# For Code Abdomen/Rec reports, determine which codes (i.e. C1-C99, REC1-REC99) is present
followup_options = []
# flag 1 = F/U label, 0 = no F/u
followup_label = []
# Use for detecting presence of a F/U option that for labeling (i.e C3,C4,C5, REC2b, etc..)
check = False

# how many reports have act 112 text
act_112_count = 0

for ind in raw_data_copy.index:
	report = raw_data_copy['report'][ind]
	rep_cat = raw_data_copy['report_category'][ind]

	# detect and remove the Act 112 Follow-up text
	try:
		result = re.findall(act_112_regex, report)
		report = report.replace(str(result),"")
		act_112_count+=1
	except:
		# no act 112 text
		pass

	# remove F/U recc
	if rep_cat == 1:
		# Code Abdomen
		# find Code Abd text block with regex
		code_abd_text_block = re.findall(code_abd_regex, report)
		followup_text.append(str(code_abd_text_block))
		report_noFU = report.replace(str(code_abd_text_block),"")	#delete the Code Abd recommendation text

		# Pull out the followup options (eg C1,C2,C3...) that are present in the report
		result = re.findall(code_abd_regex_coding, report)
		followup_options.append(result)

		# Flag C3/C4/C5 as followup_label=1
		codes_in_text = [item[0] for item in result]
		check = any([s for s in codes_in_text if any(xs in s for xs in code_abd_fu_codes)])

	elif rep_cat == 2:
		# Code Rec
		# Find Code Rec text block using regex
		code_rec_text_block = re.findall(code_rec_regex, report)
		followup_text.append(str(code_rec_text_block))
		report_noFU = report.replace(str(code_rec_text_block),"")	#delete the recommendation text
		
		# Pull out the followup options (eg REC0,REC1,REC2b...) that are present in the report
		result = re.findall(code_rec_regex_coding, report)
		followup_options.append(result)

		# Flag REC2b/3/3a/3b/3c/4/5 as followup_label=1
		codes_in_text = [item[0] for item in result]
		check = any([s for s in codes_in_text if any(xs in s for xs in code_rec_fu_codes)])

	else:
		# This text report is neither Code Abdomen or Code Rec category
		report_noFU = report
		followup_text.append("")
		followup_options.append("")
		check = False

	if check:
		followup_label.append(1)
	else:
		followup_label.append(0)

	# clean and tokenize the RAW report text
	# report_clean = clean_text(report)
	# reports_clean.append(report)
	# report_clean_tokenized = word_tokenize(report_clean)
	# reports_clean_tokenized.append(report_clean_tokenized)
	
	# clean and tokenize the report text after CodeAbdomen/CodeRec text has been removed
	report_clean_noFU = clean_text(report_noFU)
	reports_clean_noFU.append(report_clean_noFU)
	report_clean_tokenized_noFU = word_tokenize(report_clean_noFU)
	reports_clean_tokenized_noFU.append(report_clean_tokenized_noFU)

# add new column consisting of the clean tokenized text (pre-stemming)
# This applies to all reports (with or without FU reccs)
proc_reports = raw_data_copy
proc_reports['followup_text'] = followup_text
proc_reports['followup_options'] = followup_options

# copies with FU Code Abd/Rec text
# proc_reports['report_clean'] = reports_clean
# proc_reports['report_clean_tokenized'] = reports_clean_tokenized

proc_reports['report_clean_noFU'] = reports_clean_noFU
proc_reports['report_clean_tokenized_noFU'] = reports_clean_tokenized_noFU
proc_reports['fu_label'] = followup_label

# Apply porter stemming
# https://stackoverflow.com/questions/37443138/python-stemming-with-pandas-dataframe
ps = PorterStemmer()
# proc_reports['report_clean_tokenized_stemmed'] = proc_reports['report_clean_tokenized'].apply(lambda x: [ps.stem(y) for y in x])
# proc_reports['report_clean_tokenized_stemmed_FU'] = proc_reports['report_clean_tokenized'].apply(lambda x: [ps.stem(y) for y in x])
proc_reports['report_clean_tokenized_stemmed_noFU'] = proc_reports['report_clean_tokenized_noFU'].apply(lambda x: [ps.stem(y) for y in x])

# Check how many reports that had FU are not captured by regex (column: followup_text)
# we lose Followup text when the format in the report does not exactly match that which the regex looks for
missing_abd = int(proc_reports[(proc_reports['followup_text'] == "[]") & (proc_reports['report_category'] == 1)].count()[0])
missing_rec = int(proc_reports[(proc_reports['followup_text'] == "[]") & (proc_reports['report_category'] == 2)].count()[0])

# Percentage captured reports - used to refine our regex and get the # of missing reports to an acceptable level
percent_cap_abd	=	100*round((report_category.count(1)-missing_abd)/report_category.count(1),3)
percent_cap_rec	=	100*round((report_category.count(2)-missing_rec)/report_category.count(2),3)
print("\nPercent of Code Abdomen / Code Rec text blocks captured by regex:\n\tCode Abdomen\t{} \n\tCode Rec\t{}".format(percent_cap_abd, percent_cap_rec))


# TODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODO
# drop rows where we fail to capture the FU recc:
proc_reports.drop(proc_reports[(proc_reports['followup_text'] == "[]") & (proc_reports['report_category'] == 1)].index, inplace=True)
proc_reports.drop(proc_reports[(proc_reports['followup_text'] == "[]") & (proc_reports['report_category'] == 2)].index, inplace=True)

# post-drop stats
followup_label = list(proc_reports['fu_label'])
d = Counter(followup_label)
print("\nFull Dataset Statistics:")
print("\nNumber of reports\n\tfollow-up\t{} \n\tNon-followup\t{}".format(d[1], d[0]))


# print("\nSaving stage1_proc_data.csv...")
# Save as CSV for manual inspection
# proc_reports.to_csv(os.path.join(cwd,'data/processed_data/stage1_proc_data.csv'))
# proc_reports_trunc = proc_reports[['idx','followup_text','followup_options','fu_label']]
# proc_reports_trunc.to_csv(os.path.join(cwd,'data/processed_data/stage1_proc_data_TRUNCATED.csv'))

#######################################
# STEP 4 - Make train-test split sets #
#######################################
# 80/20 train/test split using scikitlearn
# Note, only pseudo-random split, because we want to have the same split for testing different models (Repeatable Train-Test Splits)
# https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/
print("\nGenerating 80/20 train-test split sets...")
df_train, df_test = train_test_split(proc_reports, test_size=0.20, random_state=1)


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
# STEP 5 - Save data for stage2.py (deep learning) #
####################################################
# From these files, the relevant columns for the deep learning model are:
# 'ID','report_clean_tokenized_stemmed_noFU','fu_label'
print("\nSaving train-test split sets...")
# df_train.to_csv(os.path.join(cwd,'data/processed_data/df_train.csv'), index=False, header=True, sep='\t')
# df_test.to_csv(os.path.join(cwd,'data/processed_data/df_test.csv'), index=False, header=True, sep='\t')

df_train.to_pickle(os.path.join(cwd,'data/processed_data/df_train.df'))
df_test.to_pickle(os.path.join(cwd,'data/processed_data/df_test.df'))

################################
# STEP 6 - Feature Engineering #
################################
# fastText
# train our word vector model using the reports that are flagged as report_category == 0 or 1
# DO NOT include the FU text in the training data (so use only the field report_clean_tokenized_stemmed_noFU)
training_data = [x for x in df_train['report_clean_tokenized_stemmed_noFU']]
print("\nTraining fastText model on {} reports...".format(len(training_data)))
# utilize fasttext implementation from gensim, skip-gram procedure (sg=1), 300-dimensional embeddings (vector_size=300)
# https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText
model = FastText(training_data, sg=1, min_count=1, vector_size=300)
# model = FastText(proc_reports, sg=1, min_count=1, vector_size=300)

word_vectors = model.wv
word_vectors.save(os.path.join(cwd,'data/vectors.kv'))

# store list of word vectors
vector_keywords = list(word_vectors.key_to_index.keys())
# save list
# with open(os.path.join(cwd,'data/processed_data/vector_keywords.txt'), 'w') as filehandle:
#     for listitem in vector_keywords:
#         filehandle.write('%s\n' % listitem)
with open(os.path.join(cwd,'data/processed_data/vector_keywords.pickle'), 'wb') as file:
	pickle.dump(vector_keywords, file)

# Example of how to view a word's vector, 300 length array of embeddings
# https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText.wv
# model.wv['node']
	
# https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText.save
FastText.save(model, os.path.join(cwd,'data/processed_data/fasttext_trained_model_300dim'))


#####################################
# STEP 7 - Pulling GloVe embeddings #
#####################################
if use_glove == 1:
	print("\nPulling GloVe embeddings...\n")
	# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
	# Use Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)
	glove_filename = 'data/glove_models/glove.840B.300d.txt'
	# glove_model = KeyedVectors.load_word2vec_format(os.path.join(cwd,glove_filename), binary=False, no_header=True)
	glove_model = KeyedVectors.load_word2vec_format(os.path.join(cwd,glove_filename), binary=False, no_header=True)

	# find the glove embeddings for our list of words (vector_keywords) and pull their vectors, store in similar format to our fasttext model for concat
	# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

	# init a matrix (300 cols for our words present in dataset
	# out-of-vocab (OOV) words missing from GloVe corpus will just be a vector of zeros
	num_missing_words=0
	glove_missing_words = []
	concat_embedding_matrix = zeros((len(vector_keywords), 600))
	i=0
	for word in vector_keywords:
		try:
			v2 = glove_model[word]
			if v2 is not None:
				v1 = word_vectors[i]
				concat_embedding_matrix[i] = np.concatenate((v1,v2))
		except:
			glove_missing_words.append(word)
			num_missing_words+=1
		i+=1
	print("Percentage of training corpus missing from GloVe:\t{}".format(round(100*num_missing_words/len(vector_keywords),3)))
	
	# PCA
	# https://theslaps.medium.com/using-pca-to-help-visualize-word-embeddings-sklearn-matplotlib-6f681979fc95
	pca = PCA(n_components=300)
	pca_embedding_data = pca.fit_transform(concat_embedding_matrix)
	# Save post-PCA word embeddings
	# https://numpy.org/doc/stable/reference/generated/numpy.save.html
	outpath = os.path.join(cwd,'data/processed_data/fasttext-glove_pca-proc_embedding_data')
	np.save(outpath, pca_embedding_data, allow_pickle=False)

	# save copy of list of the missing words (just for inspection to see how BioBERT handles these missing words)
	with open(os.path.join(cwd,'data/processed_data/glove_missing_words.txt'), 'w') as f:
		for item in glove_missing_words:
			f.write("%s\n" % item)
else:
	pass

#######################################
# STEP 8 - Pulling BioBERT embeddings #
#######################################
if use_biobert == 1:
	# Using biobert_embeddings package (with some slight modifications to embeddings.py)
	# https://github.com/Overfitter/biobert_embedding

	# Note, another package to access BioBERT is transformer
	# https://stackoverflow.com/questions/58518980/extracting-fixed-vectors-from-biobert-without-using-terminal-command

	print("\nPulling BioBERT embeddings...\n")
	biobert = BiobertEmbedding()

	# Access embeddings as follows:
	# word_embeddings = biobert.word_vector(text) 
	concat_embedding_matrix = zeros((len(vector_keywords), 1068))
	num_missing_words=0
	biobert_missing_words=[]
	i=0
	for word in vector_keywords:
		try:
			v2 = np.asarray(biobert.word_vector(vector_keywords[i])[0])
			if v2 is not None:
				v1 = word_vectors[i]
				concat_embedding_matrix[i] = np.concatenate((v1,v2))
		except:
			num_missing_words+=1
			biobert_missing_words.append(word)
		i+=1
	print("Percentage of training corpus missing from BioBERT:\t{}".format(round(100*num_missing_words/len(vector_keywords),3)))

	# PCA
	# https://theslaps.medium.com/using-pca-to-help-visualize-word-embeddings-sklearn-matplotlib-6f681979fc95
	pca = PCA(n_components=300)
	pca_embedding_data = pca.fit_transform(concat_embedding_matrix)
	# Save post-PCA word embeddings
	# https://numpy.org/doc/stable/reference/generated/numpy.save.html
	outpath = os.path.join(cwd,'data/processed_data/fasttext-biobert_pca-proc_embedding_data')
	np.save(outpath, pca_embedding_data, allow_pickle=False)
else:
	pass


print("\n------END STAGE1 SCRIPT PRINT OUTPUT------\n\n")