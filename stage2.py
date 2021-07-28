# analysis pipeline 1.0
# Stage 2 script
# Written by Nikhil Goyal, University of Pennsylvania Perelman School of Medicine, 2021

# Compatibility Information
# Written for Python 3.9.5 on Windows 10
# To execute from within the cmd line, run:
# 	python stage2.py

# written: 7/28/2021

import os
import sys

import pandas as pd
from keras.models import Sequential
from keras import layers

def main():
    # STEP 5 - Deep Learning Model
    # https://realpython.com/python-keras-text-classification/#keras-embedding-layer

    vocab_size=len(vector_keywords)
    maxlen=100
    embedding_dim = 600
    keras_model = Sequential()
    keras_model.add(layers.Embedding(	input_dim=vocab_size,
                                        output_dim=embedding_dim,
                                        input_length=maxlen))

    keras_model.add(layers.GlobalMaxPool1D())
    keras_model.add(layers.Dense(10, activation='relu'))
    keras_model.add(layers.Dense(1, activation='sigmoid'))
    keras_model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
    keras_model.summary()


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

if __name__ == '__main__':
	main()