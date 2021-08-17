# analysis pipeline 1.0
# Stage 3 script

# Compatibility Information
#   Written for Python 3.7 on Windows 10

# To execute from within the python interpreter, run:
#	exec(open("stage3.py").read())
# To run from command line:
# 	python stage2.py

# written: 8/18/2021

###################
# GlOBAL SETTINGS #
###################
max_seq_len = 600
embedding_dim = 300
output_labels = 2

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

from sklearn.metrics import roc_curve,roc_auc_score

from keras.preprocessing.text import Tokenizer
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Embedding, SpatialDropout1D, concatenate, Dropout, InputLayer
from keras.layers import CuDNNLSTM, LSTM, GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, EarlyStopping
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint


class RocAucEvaluation(Callback):
	def __init__(self, validation_data=(), interval=1):
		super(Callback, self).__init__()

		self.interval = interval
		self.x_test, self.y_test = validation_data

	def on_epoch_end(self, epoch, logs={}):
		if epoch % self.interval == 0:
			y_pred = self.model.predict(self.x_test, verbose=0)
			score = roc_auc_score(self.y_test, y_pred)
			print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

def plot_history(history):
	acc = history.history['binary_accuracy']
	val_acc = history.history['val_binary_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	x = range(1, len(acc) + 1)

	plt.figure(figsize=(12, 5))
	plt.subplot(1, 2, 1)
	plt.plot(x, acc, 'b', label='Training acc')
	plt.plot(x, val_acc, 'r', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.subplot(1, 2, 2)
	plt.plot(x, loss, 'b', label='Training loss')
	plt.plot(x, val_loss, 'r', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.savefig('training_acc_loss.png')

def plot_roc_curve(fpr,tpr): 
	plt.plot(fpr,tpr)
	plt.axis([0,1,0,1])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.savefig('ROC.png')


class Histories(Callback):
	# https://stackoverflow.com/a/52206330
	def on_train_begin(self,logs={}):
		self.losses = []
		self.accuracies = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		self.accuracies.append(logs.get('acc'))


################
# READ IN DATA #
################
cwd = os.getcwd()

with open(os.path.join(cwd,'data/processed_data/vector_keywords.pickle'), 'rb') as file:    
	vector_keywords = pickle.load(file)
vocab_size=len(vector_keywords)

df_train = pd.read_pickle(os.path.join(cwd,'data/processed_data/df_train.df'))
df_test = pd.read_pickle(os.path.join(cwd,'data/processed_data/df_test.df'))

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


#####################
# CREATE LSTM MODEL #
#####################
batch_size=128
epochs=5
patience=1
lr=0.01
inputlayer2=300
lstm_num=32

opt = SGD(lr=lr)
# model.compile(loss = "categorical_crossentropy", optimizer = opt)

histories = Histories()

embedding_matrix = np.load(os.path.join(cwd,'data/processed_data/fasttext-biobert_pca-proc_embedding_data.npy'))
model = Sequential()
model.add(InputLayer((max_seq_len,)))
# model.add(InputLayer((inputlayer2,)))
# model.add(Input(shape=(max_seq_len,)))
# model.add(Input(inputlayer2,))
model.add(Embedding(vocab_size, embedding_dim, input_length=max_seq_len, weights=[embedding_matrix], trainable=False))
# model.add(SpatialDropout1D(0.5))
# requirements to use CuDNNLSTM, https://github.com/tensorflow/tensorflow/issues/30745
model.add(Bidirectional(LSTM(lstm_num, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(4,activation='sigmoid'))
# model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
model.summary()

es_callback = EarlyStopping(monitor='val_loss', patience=patience)

# https://stackoverflow.com/questions/61706535/keras-validation-loss-and-accuracy-stuck-at-0
# history = model.fit(x_train, y_train, batch_size=256, epochs=30, validation_split=0.3, callbacks=[es_callback], shuffle=False, validation_data=[x_test,y_test])


# history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.3, callbacks=[es_callback], shuffle=False, validation_data=(x_train,y_train))
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.3, callbacks=[es_callback, histories], shuffle=False, validation_data=(x_test,y_test))
# loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))
# embedding_matrix = np.load(os.path.join(cwd,'data/processed_data/fasttext-glove_pca-proc_embedding_data.npy'))
plot_history(history)

# RocAuc = RocAucEvaluation(validation_data=(x_test, y_test), interval=1)


# RocAuc = RocAucEvaluation(validation_data=(x_test, y_test), interval=1)

# hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=[RocAuc], verbose=2)


# ROC
# https://androidkt.com/get-the-roc-curve-and-auc-for-keras-model/
# y_val_cat_prob = model.predict_proba(x_test)
# fpr , tpr , thresholds = roc_curve(y_test , y_val_cat_prob)
# auc_score = roc_auc_score(y_test, y_val_cat_prob)

####################
# CREATE GRU MODEL #
####################
""" 
inp = Input(shape=(max_seq_len, ))
x = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix])(inp)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(GRU(80, return_sequences=True))(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
conc = concatenate([avg_pool, max_pool])
outp = Dense(6, activation="sigmoid")(conc)

model = Model(inputs=inp, outputs=outp)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# RocAuc = RocAucEvaluation(validation_data=(x_test, y_test), interval=1)


hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=2)
"""
# keras_model.add(layers.Embedding(	input_dim=vocab_size,
#                                     output_dim=embedding_dim,
#                                     input_length=maxlen))

# keras_model.add(layers.GlobalMaxPool1D())
# keras_model.add(layers.Dense(10, activation='relu'))
# keras_model.add(layers.Dense(1, activation='sigmoid'))
# keras_model.compile(optimizer='adam',
#                     loss='binary_crossentropy',
#                     metrics=['accuracy'])
# keras_model.summary()



# https://realpython.com/python-keras-text-classification/#keras-embedding-layer

# vocab_size=len(vector_keywords)
# maxlen=600
# embedding_dim = 300
# keras_model = Sequential()
# keras_model.add(layers.Embedding(	input_dim=vocab_size,
#                                     output_dim=embedding_dim,
#                                     input_length=maxlen))

# keras_model.add(layers.GlobalMaxPool1D())
# keras_model.add(layers.Dense(10, activation='relu'))
# keras_model.add(layers.Dense(1, activation='sigmoid'))
# keras_model.compile(optimizer='adam',
#                     loss='binary_crossentropy',
#                     metrics=['accuracy'])
# keras_model.summary()


# Train
# history = model.fit(X_train, y_train,
#                     epochs=50,
#                     verbose=False,
#                     validation_data=(X_test, y_test),
#                     batch_size=10)
# loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))
# plot_history(history)
