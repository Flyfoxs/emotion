import numpy as np
from file_cache.utils.util_log import *
from code_felix.core.config import *
from functools import lru_cache
import pandas as pd
from file_cache.cache import file_cache

def compute_ngrams(word, min_n, max_n):
    # BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
    extended_word = word
    ngrams = []
    for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
        for i in range(0, len(extended_word) - ngram_length + 1):
            ngrams.append(extended_word[i:i + ngram_length])
    return list(set(ngrams))


def wordVec(word, wv_from_text, min_n=1, max_n=3):

    '''
    ngrams_single/ngrams_more,主要是为了当出现oov的情况下,最好先不考虑单字词向量
    '''
    # 确认词向量维度
    word_size = wv_from_text.wv.syn0[0].shape[0]
    # 计算word的ngrams词组
    ngrams = compute_ngrams(word, min_n=min_n, max_n=max_n)
    # 如果在词典之中，直接返回词向量
    if word in wv_from_text.wv.vocab.keys():
        return wv_from_text[word]
    else:
        # 不在词典的情况下
        word_vec = np.zeros(word_size, dtype=np.float32)
        ngrams_found = 0
        ngrams_single = [ng for ng in ngrams if len(ng) == 1]
        ngrams_more = [ng for ng in ngrams if len(ng) > 1]
        # 先只接受2个单词长度以上的词向量
        for ngram in ngrams_more:
            if ngram in wv_from_text.wv.vocab.keys():
                word_vec += wv_from_text[ngram]
                ngrams_found += 1
                # print(ngram)
        # 如果，没有匹配到，那么最后是考虑单个词向量
        if ngrams_found == 0:
            for ngram in ngrams_single:
                if ngram in wv_from_text:
                    word_vec += wv_from_text[ngram]
                    ngrams_found += 1
        if word_vec.any():
            return word_vec / max(1, ngrams_found)
        else:
            #raise KeyError('all ngrams for word %s absent from model' % word)
            #logger.warning(f'Can not find key for {word}')
            return None

@timed()
@lru_cache()
def load_embedding(path):
    import gensim
    word2vec_model = path
    wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=False)
    return wv_from_text


@timed()
@lru_cache()
def load_embedding_dict(path):
    embedding_index = {}
    f = open(path, encoding='utf8')
    for index, line in enumerate(f):
        if index == 0:
            continue
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()

    return embedding_index


import jieba
from functools import lru_cache


@timed()
@lru_cache()
def load_jeba_userdict():
    jieba.load_userdict('./input/jieba.txt')


@timed()
def get_word_freqs():
    import collections
    word_freqs = collections.Counter()

    for file in [train_file, test_file]:
        df = pd.read_csv(file, encoding='gb18030', delimiter='\t', header=None)
        for sentence in df.iloc[:, 2].values.tolist():
            for word in jieba.cut(str(sentence), cut_all=False):
                word_freqs[word] += 1
    df = pd.DataFrame({'word': list(word_freqs.keys()), 'freqs': list(word_freqs.values()), })
    df.sort_values('freqs', ascending=False, inplace=True)

    df.reset_index(drop=True, inplace=True)
    df['id'] = df.index + 2

    df = df.append({'freqs': 0, 'word': 'PAD', 'id': 0}, ignore_index=True)
    df = df.append({'freqs': 0, 'word': 'UNK', 'id': 1}, ignore_index=True)

    return df.sort_values('id')


@timed()

# def get_id2vec(fname="./output/mini.kv"):
#     from gensim.models import KeyedVectors
#     wv_from_text = KeyedVectors.load(fname, mmap='r')
#     wv_from_text.init_sims()
def get_id2vec(word2vec_model="./output/mini_v6.txt"):
    import gensim
    wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=False)
    wv_from_text.init_sims()
    ordered_vocab = [(term, voc.index, voc.count) for term, voc in wv_from_text.wv.vocab.items()]
    # sort by the term counts, so the most common terms appear first
    ordered_vocab = sorted(ordered_vocab, key=lambda k: -k[2])

    # unzip the terms, integer indices, and counts into separate lists
    ordered_terms, term_indices, term_counts = zip(*ordered_vocab)
    # print(ordered_terms)
    # create a DataFrame with the food2vec vectors as data,
    # and the terms as row labels
    word_vectors = pd.DataFrame(wv_from_text.wv.syn0norm[term_indices, :], index=ordered_terms)
    word_vectors.index.name = 'word'
    return word_vectors.reset_index()



@file_cache()
def get_word_id_vec(version='1'):
    freqs = get_word_freqs()
    word_vec = get_id2vec()
    return pd.merge(freqs, word_vec, how='left', on='word')


from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))