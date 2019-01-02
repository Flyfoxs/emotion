import sys
import os

from keras.callbacks import EarlyStopping

from code_felix.core.feature import *
from file_cache.utils.util_log import *

import matplotlib.pyplot as plt

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


#Loading the dataset
dataset = pd.read_csv(train_file, encoding='gb18030', delimiter='\t', header=None)

input_sentences = [list(jieba.cut(str(text), cut_all=False))  for text in dataset.iloc[:, 2].values.tolist()]
labels = dataset.iloc[:, 3].values.tolist()

word_id_vec =  get_word_id_vec('02')
word2id = dict( word_id_vec.apply(lambda row: (row['word'], row['id']), axis=1).values )
logger.debug(f'Word length:{len(word2id)}')


embedding_weights = word_id_vec.iloc[:, -vector_size:].fillna(0).values
#embedding_weights[10]

# Initialize word2id and label2id dictionaries that will be used to encode words and labels

label2id = dict()

max_words = 0  # maximum number of words in a sentence

# Construction of word2id dict
for sentence in input_sentences:
    #     for word in sentence:
    #         # Add words to word2id dict if not exist
    #         if word not in word2id:
    #             word2id[word] = len(word2id)
    #     # If length of the sentence is greater than max_words, update max_words
    #     sentence = list(sentence)
    #     logger.debug(f'{len(sentence)} : {sentence}')
    if len(sentence) > max_words:
        max_words = len(sentence)
        logger.debug(f'max_words={max_words}')

# Construction of label2id and id2label dicts
label2id = {l: i for i, l in enumerate(set(labels))}
id2label = {v: k for k, v in label2id.items()}
id2label


import keras

# Encode input words and labels
X = [[word2id[word] for word in sentence] for sentence in input_sentences]
Y = [label2id[label] for label in labels]

# Apply Padding to X
from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(X, max_words)

# Convert Y to numpy array
Y = keras.utils.to_categorical(Y, num_classes=len(label2id))

# Print shapes
print("Shape of X: {}".format(X.shape))
print("Shape of Y: {}".format(Y.shape))


def train(trainable, weights):
    embedding_dim = 100  # The dimension of word embeddings

    # Define input tensor
    sequence_input = keras.Input(shape=(max_words,), dtype='int32')

    # Word embedding layer
    embedded_inputs = keras.layers.Embedding(len(word2id),
                                             vector_size,
                                             input_length=max_words,
                                             weights=weights,
                                             trainable=trainable,
                                             )(sequence_input)

    # Apply dropout to prevent overfitting
    embedded_inputs = keras.layers.Dropout(0.2)(embedded_inputs)

    # Apply Bidirectional LSTM over embedded inputs
    lstm_outs = keras.layers.wrappers.Bidirectional(
        keras.layers.LSTM(embedding_dim, return_sequences=True)
    )(embedded_inputs)

    # Apply dropout to LSTM outputs to prevent overfitting
    lstm_outs = keras.layers.Dropout(0.2)(lstm_outs)

    # Attention Mechanism - Generate attention vectors
    input_dim = int(lstm_outs.shape[2])
    permuted_inputs = keras.layers.Permute((2, 1))(lstm_outs)
    attention_vector = keras.layers.TimeDistributed(keras.layers.Dense(1))(lstm_outs)
    attention_vector = keras.layers.Reshape((max_words,))(attention_vector)
    attention_vector = keras.layers.Activation('softmax', name='attention_vec')(attention_vector)
    attention_output = keras.layers.Dot(axes=1)([lstm_outs, attention_vector])

    # Last layer: fully connected with softmax activation
    fc = keras.layers.Dense(embedding_dim, activation='relu')(attention_output)
    output = keras.layers.Dense(len(label2id), activation='softmax')(fc)

    # Finally building model
    model = keras.Model(inputs=[sequence_input], outputs=output)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer='adam')

    # Print model summary
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', verbose=1,
                               patience=3,
                               )

    history = model.fit(X, Y,
                        callbacks=[early_stop],
                        epochs=30, batch_size=64, validation_split=0.2, shuffle=True)

    best_epoch_loss = np.array(history.history['val_loss']).argmin() + 1
    best_score_loss = round(np.array(history.history['val_loss']).min(), 5)

    best_epoch_acc = np.array(history.history['val_acc']).argmax() + 1
    best_score_acc = round(np.array(history.history['val_acc']).max(), 5)

    logger.debug(f'Result summary with paras:trainable:{trainable}, weight:{False if weights is None else True }, '
                 f'epoch_acc:{best_epoch_acc}, score_acc:{best_score_acc}, '
                 f'epoch_loss:{best_epoch_loss}, score_loss:{best_score_loss}, ')


if __name__ == '__main__':
    train(False, [embedding_weights])

    train(True, [embedding_weights])

    train(False, None)
