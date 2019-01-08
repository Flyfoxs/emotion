import sys
import os

from keras import models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from code_felix.core.feature import *
from file_cache.utils.util_log import *
from code_felix.core.config import *

import matplotlib.pyplot as plt

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from file_cache.utils.other import replace_invalid_filename_char

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
def train(drop_out, max_words=221 , embedding_dim =100, max_epochs=40, trainable = False):
    input_args = get_mini_args(locals())

    model_meta_file = f'./output/checkpoint/meta_{input_args}.csv'
    model_list, start_epoch = get_model_list(model_meta_file, drop_out, max_words, embedding_dim, trainable)

    for epoch_current in range(1+start_epoch, max_epochs+1):
        score_list = []
        for sn, model in enumerate(model_list):
            X, Y, test = get_feature(max_words)

            reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=2, verbose=1)

            history = model.fit(X, Y,
                                #callbacks=[reduce],
                                epochs=1,
                                batch_size=64,
                                #validation_split=0.2,
                                shuffle=True)

            logger.debug(f'paras:{input_args}, history:{history}')

            best_epoch_loss = np.array(history.history['loss']).argmin() + 1
            best_score_loss = round(np.array(history.history['loss']).min(), 5)

            best_epoch_acc = np.array(history.history['acc']).argmax() + 1
            best_score_acc = round(np.array(history.history['acc']).max(), 5)

            best_epoch_f1 = np.array(history.history['f1']).argmax() + 1
            best_score_f1 = round(np.array(history.history['f1']).max(), 5)

            logger.debug(f'Result summary with epoch:{epoch_current}, model#{sn}, paras:{input_args}, '
                         # f'epoch_f1:{best_epoch_f1}, '
                         f'f1:{best_score_f1:.5f}, '
                         # f'epoch_acc:{best_epoch_acc}, '
                         f'acc:{best_score_acc:.5f}, '
                         # f'epoch_loss:{best_epoch_loss}, '
                         f'loss:{best_score_loss:.5f}, ')
            score_list.append( best_score_f1 )
        score_avg = np.array(score_list).mean()
        score_std = np.array(score_list).std()
        logger.debug(f'poch:{epoch_current}, The avg:{score_avg:0.5f}, std:{score_std:0.5f} for epoch:{epoch_current}: {score_list}')

        save_model(model_meta_file, model_list, epoch_current, drop_out, max_words, embedding_dim, trainable)
        if score_avg > 0.945:
            gen_sub(model_list, test, f'{score_avg:0.5f}_{input_args}_cur_epoch:{epoch_current}')
        else:
            logger.debug(f'The socore avg is {score_avg}, no need to gen sub')



    # if os.path.exists(tmp_model):
    #     os.remove(tmp_model)

    return best_score_acc

@timed()
def save_model(meta_path, model_list,  epoch, *args):
    model_meta = _get_model_mete(meta_path)
    if model_meta is None:
        model_meta = pd.DataFrame(np.empty((len(model_list), 2)), columns=['model_path','epoch'])
        model_meta.epoch = 0
        for i in range(len(model_list)):
            model_meta.loc[i,'model_path'] = replace_invalid_filename_char(f'./output/checkpoint/model_{i}_{get_mini_args(args)}.hdf5')

    for model, path in zip(model_list, model_meta.model_path.values):
        model.save(path, overwrite=True)

    logger.debug(f'Model is save base on df:{model_meta}')
    model_meta.epoch = epoch
    model_meta.to_csv(meta_path, index=False)


def _get_model_mete(meta_path):
    if os.path.exists(meta_path):
        df = pd.read_csv(meta_path)
        return df
    else:
        return None

def get_model_list(meta_path, *args):
    model_meta = _get_model_mete(meta_path)

    if model_meta is None:

        model_list = [ ]
        for _ in range(5):
            model_list.append(get_model(*args))
        logger.debug(f'No existing models are found, inistal {len(model_list)} models')
        return model_list, 0
    else:
        model_list = [ models.load_model(model_path,custom_objects={'f1': f1})  for model_path in model_meta.model_path]
        logger.debug(f'Load existing models base on {model_meta}')
        return model_list, int(model_meta.at[0,'epoch'])


@timed()
def gen_sub(model_list, test, comments):
    from keras import models
    import socket
    host_name = socket.gethostname().split('-')[-1]
    predict_sum = np.zeros((len(test),3))
    # score_sum = 0
    logger.debug(f'There are {len(model_list)} models need to average')
    for model in model_list:
        # model = models.load_model(model_file,custom_objects={'f1': f1})
        #model.predict(test)
        tmp = model.predict(test)

        logger.debug(f'temp:{tmp[0]}')
        predict_sum += tmp
        # score_sum += score

    predict_propability = predict_sum/len(model_list)
    # score = score_sum/len(model_list)
    logger.debug((f'Final:{predict_propability[0]}'))

    predict = np.argmax(predict_propability, axis=1)
    sub = pd.DataFrame(model.predict(test), columns=range(3))
    sub['emotion'] = sub.idxmax(axis=1)
    sub['id'] = range(1, len(predict)+1)


    file_name = replace_invalid_filename_char(f'./output/sub/emo_avg{len(model_list)}_{comments}__{host_name}.csv')

    #SUB FILE
    logger.debug(f'Save sub file to :{file_name}')
    sub[['id', 'emotion']].to_csv(file_name, header=False, index=False)

    #LEVEL1 FILE
    file_name = replace_invalid_filename_char(f'./output/1level/emo_avg{len(model_list)}_{comments}_{host_name}.hd5')
    sub.to_hdf(file_name, header=False, index=False, key='test')



def get_model(drop_out, max_words, embedding_dim, trainable):
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
                                             trainable=trainable,
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
    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy",f1], optimizer=adam)
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
    if 1 in args:
        train(0.2)
    if 2 in args:
        train(0.5)
    if 3 in args:
        train(0.4)





    # for i in range(5):
    #     #embedding_dim
    #     if args is None or 1 in args:
    #             for embedding_dim in range(80, 150, 20):
    #                 train(0.5, 221, embedding_dim)
    #
    #     #Drop out
    #     if args is None or 2 in args:
    #         for drop_out in np.arange(0.2, 0.6, 0.1):
    #             train(round(drop_out, 1), 221)
    #
    #     # #max_words
    #     # if args is None or 3 in args:
    #     #         for max_words in [200, 221, 240, ]:
    #     #             train(0.2, max_words, embedding_dim=100)
    #
    #     #Drop out
    #     if args is None or 4 in args:
    #         for drop_out in [0.3,0.2]:
    #             for _ in range(4):
    #                 train(round(drop_out, 1), 221)
    #         # for drop_out in [0.5]:
    #         #     train(round(drop_out, 1), 221)
    #         # for drop_out in [0.5]:
    #         #     train(round(drop_out, 1), 221)
    #
    #
    #
    #
    #
    #



