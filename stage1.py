# analysis pipeline 1.0
# Stage 1 script
# Written by Nikhil Goyal, University of Pennsylvania Perelman School of Medicine, 2021

# Compatibility Information
# 	Written for Python 3.7 on Windows 10

# To execute from within the python interpreter, run:
#	exec(open("stage1.py").read())
# To run from command line:
# 	python stage1.py

###################
# GlOBAL SETTINGS #
###################
use_glove = 1
use_biobert = 1

###########
# IMPORTS #
###########
import os
import re
import string

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
from biobert_embedding.embedding import BiobertEmbedding

# PCA on word embeddings
from sklearn.decomposition import PCA

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


######################################################
# STEP 2 - Preliminary identification of F/U recommendation and statistics #
######################################################
# 1. Detect presence of a Code Abdomen / Code Rec entry / ABDOMEN-PELVIS for puposes of stripping from the report
# 2. Count number of reports with incidental findings (based on Code Abdomen, Code Rec, other)
# 3. Pull the entries in the findings (ex the C1-C99, REC0-REC99 codes) and store as strings in a list

# Variables for data aggregation and statistics
# use followup_type_list to store 0 for Code Abdomen, 1 for Code Rec - used to determine if these reports contain F/U and if we can use them as 'labeled' training data
followup_type_list = []
incidental_finding_count = 0
raw_data_copy = raw_data

for ind in raw_data_copy.index:
	report = raw_data_copy['report'][ind]
	if "FOCAL_MASS_SUMMARY" in report:
		incidental_finding_count+=1
		# classify report as Code Abdomen
		followup_type_list.append(1)
	elif "NON-EMERGENT ACTIONABLE FINDINGS" in report:
		incidental_finding_count+=1
		# classify report as Code Rec
		followup_type_list.append(2)
	elif "START INTERVAL ONCOLOGIC RESPONSE ASSESSMENT (ABDOMEN-PELVIS)" in report:
		# Code Cancer - THIS IS NOT A F/U RECOMMENDATION
		# Drop these reports because they may skew our model since all these patients have known malignancy
		followup_type_list.append(-1)
	else:
		# case where F/U not present
		followup_type_list.append(-2)

# make new column with the followup_type as new column
raw_data_copy['fu_type'] = followup_type_list

# Basic statistics
print("\n\n------BEGIN STAGE1 SCRIPT PRINT OUTPUT------\n")
print("numer of Code Abdomen | Code Rec | Abdomen-Pelvis | None:", followup_type_list.count(1), followup_type_list.count(2), followup_type_list.count(-1), followup_type_list.count(-2))
print("total number NON-CodeAbd/NON-CodeRec", followup_type_list.count(-1)+followup_type_list.count(-2))

# Drop all Code Cancer reports (flagged 'fu_type' == -1)
raw_data_copy.drop(raw_data_copy[raw_data_copy['fu_type'] == -1].index, inplace=True)

###############################################################
# STEP 3 - Obtain fundamental semantic content of each report #
# note, functional as of 7/17/21
###############################################################
# Note 7/20/2021 - fastText has a tokenizer, possible option to use that instead of nltk
# Obtaining semantic content can be done one of two ways
# a. (Per Lou 2020, JDI) lower case, remove punctuation/symbols/numbers, tokenize, remove stop words, porter stem (PRESENT CODE USES THIS)
# b. lower case, tokenize, remove punctuation/symbols/numbers, remove stop words, porter stem

# Codes to be used for Follow-up label
code_abd_fu_codes = ['C3','C4','C5']
code_rec_fu_codes = ['REC2b','REC3','REC3a','REC3b','REC3c','REC4','REC5']

# Code Abdomen
# regex uses the negative lookahead \}(?!,) to find closing brace not followed by a commma
code_abd_regex = re.compile("(?sim)FOCAL_MASS_SUMMARY.*?\}(?!,)")
# regex to capture specific Code Abdomen option to determine if true F/U present
# call with re.findall to get all options listed
code_abd_regex_coding = re.compile("(?sim)(\{C(1|2|3|4|5|6|7|99):)")

# Code Rec
# This regex is used to capture the entire Code Rec block for deletion from the text
# code_rec_regex = re.compile("(?sim)^NON-EMERGENT ACTIONABLE FINDINGS\s?\n^((Recommendation:[\w\s\d.]+(\[REC[\d]+[\w]?\][\w\s\d]?)+)+)")
code_rec_regex = re.compile("(?ims)(NON-EMERGENT ACTIONABLE FINDINGS)[\w\n\s\d,.!?\\\/-]*(Recommendation:[\w\s\d,.!?\\\/\-();:\[\]\{\}]*(\[REC[\d]+[\w]?\])[\n\s\d,.!?\\\/-]*\n*)")
# Regex to capture the REC code associated with the Code Rec text
# call with re.findall to get all options listed
code_rec_regex_coding = re.compile("(?sim)(\[REC(0|1|2a|2b|3a|3b|3c|4|5|99)\])")

# Act 112 Notification
# Format of this block of text:
# 	FOLLOW-UP NOTICE: blah blah blah blah. [FOL3M]
# Detect and redact using regex to capture:
# 	FOLLOW-UP NOTICE: ......... [FOLxx]
# act_112_regex = re.compile("(?ims)(FOLLOW-UP NOTICE:[\n\w\s\d,.!?\\\/\-();:\[\]\{\}]*\[FOL[\d\w].*(?=\]))")
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

# for sake of curiosity, how many reports have act 112 text
act_112_count = 0

# flag 1 = F/U label, 0 = no F/u
followup_label = []
# Use for detecting presence of a F/U option that for labeling (i.e C3,C4,C5, REC2b, etc..)
check = False

for ind in raw_data_copy.index:
	report = raw_data_copy['report'][ind]
	followup_type = raw_data_copy['fu_type'][ind]

	# detect and remove the Act 112 Follow-up text
	try:
		result = re.findall(act_112_regex, report)
		report = report.replace(str(result),"")
		act_112_count+=1
	except:
		# no act 112 text
		pass

	# remove F/U recc
	if followup_type == 1:
		# code abdomen
		result = re.findall(code_abd_regex, report)
		followup_text.append(str(result))
		report_noFU = report.replace(str(result),"")	#delete the recommendation text
		result = re.findall(code_abd_regex_coding, report)
		followup_options.append(result)
		
		# look at the Code Abdomen codes and determine if C3, C4, or C5 (all will be flagged as F/U)
		# https://www.codegrepper.com/code-examples/python/check+if+a+list+contains+any+item+from+another+list+python
		# Check any element of code_abd_fu_codes in result
		check = any(item in result for item in code_abd_fu_codes)

	elif followup_type == 2:
		# code rec
		result = re.findall(code_rec_regex, report)
		followup_text.append(str(result))
		report_noFU = report.replace(str(result),"")	#delete the recommendation text
		result = re.findall(code_rec_regex_coding, report)
		followup_options.append(result)

		# look at Code Rec codes and determine if REC2b/3*/4/5 present (all will be flagged as F/U)
		check = any(item in result for item in code_rec_fu_codes)

	else:
		# report is neither Code Abdomen or Code Rec, so we consider it to NOT contain a followup recommendation
		report_noFU = report
		followup_text.append("")
		followup_options.append("")
		pass

	if check:
		followup_label.append(1)
	else:
		followup_label.append(0)

	# clean and tokenize the RAW report text (i.e. FU text has not been removed)
	report_clean = clean_text(report)
	reports_clean.append(report)
	report_clean_tokenized = word_tokenize(report_clean)
	reports_clean_tokenized.append(report_clean_tokenized)

	# If this report happens to be one that contains a FU, we need to make a copy of report with the Code Abdomen / Code Rec F/U text REMOVED
	# Note, to keep the lists the correct length, only do if followup_type > 0, otherwise these lists end up having as many rows as raw_data_copy
	report_clean_noFU = clean_text(report_noFU)
	reports_clean_noFU.append(report_clean_noFU)
	report_clean_tokenized_noFU = word_tokenize(report_clean_noFU)
	reports_clean_tokenized_noFU.append(report_clean_tokenized_noFU)


# add new column consisting of the clean tokenized text (pre-stemming)
# This applies to all reports (with or without FU reccs)
proc_reports = raw_data_copy
proc_reports['report_clean_tokenized'] = reports_clean_tokenized
proc_reports['followup_recommendation'] = followup_text
proc_reports['followup_options'] = followup_options
proc_reports['report_clean_noFU'] = reports_clean_noFU
proc_reports['report_clean_tokenized_noFU'] = reports_clean_tokenized_noFU
proc_reports['fu_label'] = followup_label

# Apply porter stemming
# https://stackoverflow.com/questions/37443138/python-stemming-with-pandas-dataframe
ps = PorterStemmer()
proc_reports['report_clean_tokenized_stemmed'] = proc_reports['report_clean_tokenized'].apply(lambda x: [ps.stem(y) for y in x])
proc_reports['report_clean_tokenized_stemmed_FU'] = proc_reports['report_clean_tokenized'].apply(lambda x: [ps.stem(y) for y in x])
proc_reports['report_clean_tokenized_stemmed_noFU'] = proc_reports['report_clean_tokenized_noFU'].apply(lambda x: [ps.stem(y) for y in x])

# Check how many reports that had FU text have no follow up text extracted (column: followup_recommendation)
# Note, it appears that we lose Followup text when the format in the report does not exactly match that which the regex looks for
# find reports missing FU text
missing_abd = int(proc_reports[(proc_reports['followup_recommendation'] == "[]") & (proc_reports['fu_type'] == 1)].count()[0])
missing_rec = int(proc_reports[(proc_reports['followup_recommendation'] == "[]") & (proc_reports['fu_type'] == 2)].count()[0])

# Percentage captured reports - used to refine our regex and get the # of missing reports to an acceptable level
percent_cap_abd	=	100*round((followup_type_list.count(1)-missing_abd)/followup_type_list.count(1),3)
percent_cap_rec	=	100*round((followup_type_list.count(2)-missing_rec)/followup_type_list.count(2),3)
print("\nPercent FU captured by regex in:\n\tCode Abdomen:{} \t|\t Code Rec:{}".format(percent_cap_abd, percent_cap_rec))

# Save as CSV for visual inspection
# cwd = os.getcwd()
proc_reports.to_csv(os.path.join(cwd,'data/processed_data/stage1_proc_data.csv'))


################################
# STEP 4 - Feature Engineering #
################################
# fastText
# train our word vector model using the reports that are flagged as fu_type == 0 or 1
# DO NOT include the FU text in the training data (so use only the field report_clean_tokenized_stemmed_noFU)
training_data = [x for x in proc_reports['report_clean_tokenized_stemmed_noFU']]
# utilize fasttext implementation from gensim, skip-gram procedure (sg=1), 300-dimensional embeddings (vector_size=300)
# https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText
model = FastText(training_data, sg=1, min_count=1, vector_size=300)
word_vectors = model.wv
word_vectors.save(os.path.join(cwd,'data/vectors.kv'))
# store list of word vectors
vector_keywords = list(word_vectors.key_to_index.keys())

# Example of how to view a word's vector, 300 length array of embeddings
# https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText.wv
# model.wv['node']
	
# https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText.save
FastText.save(model, os.path.join(cwd,'data/fasttext_trained_model_300dim'))

#####################################
# STEP 5 - Pulling GloVe embeddings #
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
	missing_words = []
	concat_embedding_matrix = zeros((len(vector_keywords), 600))
	i=0
	for word in vector_keywords:
		try:
			v2 = glove_model[word]
			if v2 is not None:
				v1 = word_vectors[i]
				concat_embedding_matrix[i] = np.concatenate((v1,v2))
		except:
			missing_words.append(word)
			num_missing_words+=1
		i+=1
	print("Percentage of training corpus missing from GloVe:\t{}".format(round(100*num_missing_words/len(vector_keywords),3)))
	
	# PCA
	# https://theslaps.medium.com/using-pca-to-help-visualize-word-embeddings-sklearn-matplotlib-6f681979fc95
	pca = PCA(n_components=300)
	pca_embedding_data = pca.fit_transform(concat_embedding_matrix)
	# Save post-PCA word embeddings
	# https://numpy.org/doc/stable/reference/generated/numpy.save.html
	outpath = os.path.join(cwd,'data/processed_data/glove_pca_embedding_data')
	np.save(outpath, pca_embedding_data, allow_pickle=False)
else:
	pass

#######################################
# STEP 6 - Pulling BioBERT embeddings #
#######################################
if use_biobert == 1:
	# Using biobert_embeddings package
	# Note, another package to access BioBERT is transformer
	# https://stackoverflow.com/questions/58518980/extracting-fixed-vectors-from-biobert-without-using-terminal-command

	print("\nPulling BioBERT embeddings...\n")
	biobert = BiobertEmbedding()

	# Access embeddings as follows:
	# word_embeddings = biobert.word_vector(text) 
	concat_embedding_matrix = zeros((len(vector_keywords), 1068))
	num_missing_words=0
	missing_words=[]
	i=0
	for word in vector_keywords:
		try:
			v2 = np.asarray(biobert.word_vector(vector_keywords[i])[0])
			if v2 is not None:
				v1 = word_vectors[i]
				concat_embedding_matrix[i] = np.concatenate((v1,v2))
		except:
			num_missing_words+=1
			missing_words.append(word)
		i+=1
	print("Percentage of training corpus missing from BioBERT:\t{}".format(round(100*num_missing_words/len(vector_keywords),3)))

	# PCA
	# https://theslaps.medium.com/using-pca-to-help-visualize-word-embeddings-sklearn-matplotlib-6f681979fc95
	pca = PCA(n_components=300)
	pca_embedding_data = pca.fit_transform(concat_embedding_matrix)
	# Save post-PCA word embeddings
	# https://numpy.org/doc/stable/reference/generated/numpy.save.html
	outpath = os.path.join(cwd,'data/processed_data/biobert_pca_embedding_data')
	np.save(outpath, pca_embedding_data, allow_pickle=False)
else:
	pass

	####################################################
	# STEP 7 - Save data for stage2.py (deep learning) #
	####################################################
	# Save tokenized, labeled data (large dataframe)
	# Save report id/label files (for training/testing)

print("\n------END STAGE1 SCRIPT PRINT OUTPUT------\n\n")