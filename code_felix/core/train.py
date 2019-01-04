import sys
import os


from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from code_felix.core.feature import *
from file_cache.utils.util_log import *
from code_felix.core.config import *

import matplotlib.pyplot as plt

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

np.random.seed(42)
import keras


@timed()
@lru_cache()
def get_feature(max_words ):
    #global word2id, label2id, X, Y
    # Loading the dataset
    dataset = pd.read_csv(train_file, encoding='gb18030', delimiter='\t', header=None)
    input_sentences = [list(jieba.cut(str(text), cut_all=False)) for text in dataset.iloc[:, 2].values.tolist()]
    labels = dataset.iloc[:, 3].values.tolist()
    word_id_vec = get_word_id_vec('02')
    word2id = dict(word_id_vec.apply(lambda row: (row['word'], row['id']), axis=1).values)
    logger.debug(f'Word length:{len(word2id)}')

    label2id = dict()
    # Construction of label2id and id2label dicts
    label2id = {l: i for i, l in enumerate(set(labels))}


    # Encode input words and labels
    X = [[word2id[word] for word in sentence] for sentence in input_sentences]
    Y = [label2id[label] for label in labels]
    # Apply Padding to X
    from keras.preprocessing.sequence import pad_sequences
      # maximum number of words in a sentence
    X = pad_sequences(X, max_words)
    # Convert Y to numpy array
    Y = keras.utils.to_categorical(Y, num_classes=len(label2id))
    # Print shapes
    print("Shape of X: {}".format(X.shape))
    print("Shape of Y: {}".format(Y.shape))

    dataset_test = pd.read_csv(test_file, encoding='gb18030', delimiter='\t', header=None)
    input_sentences = [list(jieba.cut(str(text), cut_all=False)) for text in dataset_test.iloc[:, 2].values.tolist()]
    test = [[word2id[word] for word in sentence] for sentence in input_sentences]
    test= pad_sequences(test, max_words)

    return X, Y, test




@timed()
def train(drop_out, max_words, embedding_dim =100):
    input_args = get_mini_args(locals())

    model = get_model(drop_out, max_words, embedding_dim)
    X, Y, test = get_feature(max_words)



    tmp_model = f'./output/checkpoint/emotion_{input_args}.hdf5'
    check_best = ModelCheckpoint(filepath=tmp_model,
                                monitor='val_acc',verbose=1,
                                save_best_only=True, mode='max')

    early_stop = EarlyStopping(monitor='val_loss', verbose=1,
                               patience=5,
                               )

    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                               patience=2, verbose=1)

    history = model.fit(X, Y,
                        callbacks=[early_stop,check_best, reduce],
                        epochs=30, batch_size=64, validation_split=0.2, shuffle=True)

    logger.debug(f'paras:{input_args}, history:{history}')

    best_epoch_loss = np.array(history.history['val_loss']).argmin() + 1
    best_score_loss = round(np.array(history.history['val_loss']).min(), 5)

    best_epoch_acc = np.array(history.history['val_acc']).argmax() + 1
    best_score_acc = round(np.array(history.history['val_acc']).max(), 5)

    best_epoch_f1 = np.array(history.history['val_f1']).argmax() + 1
    best_score_f1 = round(np.array(history.history['val_f1']).max(), 5)

    logger.debug(f'Result summary with paras:{input_args}, '
                 f'epoch_f1:{best_epoch_f1}, f1:{best_score_f1}, '
                 f'epoch_acc:{best_epoch_acc}, acc:{best_score_acc}, '
                 f'epoch_loss:{best_epoch_loss}, loss:{best_score_loss}, ')

    gen_sub(tmp_model,test, f'{best_score_acc}_{input_args}_{best_epoch_acc}')

    if os.path.exists(tmp_model):
        os.remove(tmp_model)

    return best_score_acc

def gen_sub(model_file, test, comments):
    from keras import models
    model = models.load_model(model_file,custom_objects={'f1': f1})
    #model.predict(test)
    predict = np.argmax(model.predict(test), axis=1)
    sub = pd.DataFrame(model.predict(test), columns=range(3))
    sub['emotion'] = sub.idxmax(axis=1)
    sub['id'] = range(1, len(predict)+1)

    from file_cache.utils.other import replace_invalid_filename_char
    file_name = replace_invalid_filename_char(f'./output/sub/emotion_{comments}.csv')

    #SUB FILE
    logger.debug(f'Save sub file to :{file_name}')
    sub[['id', 'emotion']].to_csv(file_name, header=False, index=False)

    #LEVEL1 FILE
    file_name = replace_invalid_filename_char(f'./output/1level/emotion_{comments}.hd5')
    sub.to_hdf(file_name, header=False, index=False, key='test')



def get_model(drop_out, max_words, embedding_dim):
    global vector_size
    #embedding_dim = 100  # The dimension of word embeddings
    output_dim  =3 # how many emotion need to category
    embedding_weights = get_word_id_vec('02').iloc[:, -vector_size:].fillna(0).values

    # Define input tensor
    sequence_input = keras.Input(shape=(max_words,), dtype='int32')
    # Word embedding layer
    embedded_inputs = keras.layers.Embedding(len(embedding_weights),
                                             vector_size,
                                             input_length=max_words,
                                             weights=[embedding_weights],
                                             trainable=False,
                                             )(sequence_input)
    # Apply dropout to prevent overfitting
    embedded_inputs = keras.layers.Dropout(drop_out)(embedded_inputs)
    # Apply Bidirectional LSTM over embedded inputs
    lstm_outs = keras.layers.wrappers.Bidirectional(
        keras.layers.LSTM(embedding_dim, return_sequences=True)
    )(embedded_inputs)
    # Apply dropout to LSTM outputs to prevent overfitting
    lstm_outs = keras.layers.Dropout(drop_out)(lstm_outs)
    # Attention Mechanism - Generate attention vectors
    input_dim = int(lstm_outs.shape[2])
    permuted_inputs = keras.layers.Permute((2, 1))(lstm_outs)
    attention_vector = keras.layers.TimeDistributed(keras.layers.Dense(1))(lstm_outs)
    attention_vector = keras.layers.Reshape((max_words,))(attention_vector)
    attention_vector = keras.layers.Activation('softmax', name='attention_vec')(attention_vector)
    attention_output = keras.layers.Dot(axes=1)([lstm_outs, attention_vector])
    # Last layer: fully connected with softmax activation
    fc = keras.layers.Dense(embedding_dim, activation='relu')(attention_output)
    output = keras.layers.Dense(output_dim, activation='softmax')(fc)
    # Finally building model
    model = keras.Model(inputs=[sequence_input], outputs=output)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy",f1], optimizer='adam')
    # Print model summary
    model.summary()
    return model


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        args = [int(item) for item in args]
    else:
        args = None

    logger.debug(f'The section paras is:{args}')

    for i in range(5):
        #embedding_dim
        if args is None or 1 in args:
            for max_words in [221]:
                for embedding_dim in range(80, 200, 20):
                    train(0.5, max_words, embedding_dim)

        #Drop out
        if args is None or 2 in args:
            for drop_out in np.arange(0.1, 0.7, 0.1):
                train(round(drop_out, 1), 221)

        #max_words
        if args is None or 3 in args:
                for max_words in [200, 221, 240, ]:
                    train(0.5, max_words, embedding_dim=100)

        #Drop out
        if args is None or 4 in args:
            for drop_out in [0.5, 0.6, 0.4]:
                for _ in range(5):
                    train(round(drop_out, 1), 221)
            # for drop_out in [0.5]:
            #     train(round(drop_out, 1), 221)
            # for drop_out in [0.5]:
            #     train(round(drop_out, 1), 221)









