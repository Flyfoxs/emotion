import numpy as np
import gensim
from code_felix.core.config import *
import pandas as pd

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
                word_vec += wv_from_text[ngram]
                ngrams_found += 1
        if word_vec.any():
            return word_vec / max(1, ngrams_found)
        else:
            raise KeyError('all ngrams for word %s absent from model' % word)

@timed()
def load_embedding(path):
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


@timed()
def gen_mini_embedding(full, mini, file, cut_all=True):
    input_text = pd.read_csv(file, encoding='gb18030', delimiter='\t', header=None)
    #input_text = input_text[:200]
    #input_text.head()
    jieba.load_userdict(jieba_dict)

    logger.debug(f'load jieba dict:{len(list(jieba.dt.FREQ.keys()))} from file:{jieba_dict}')

    input_text['jieba'] = input_text.iloc[:, 2].apply(lambda text: ' '.join(jieba.cut(str(text), cut_all)))
    #input_text['jieba_len'] = input_text['jieba'].apply(lambda text: len(text.split(' ')))
    for index, text in enumerate(input_text['jieba'].values):
        for word in text.strip().split(' '):
            if word in full and word not in mini:
                mini[word] = full[word]
            elif word not in full and len(word) == 1:
                # print(f'Canot find vec for:{word}')
                pass
            elif word not in full and len(word) > 1:
                print(f'Canot find vec for:{len(word)}, {word}, index:{index}')
            else:
                pass
    return mini


# compute_ngrams('搞来搞去', 1, 3)
# wordVec('搞来搞去', embed, 1, 3)



embed = load_embedding(word2vec_model)
vector_size = 200
mini = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size)
mini = gen_mini_embedding(embed, mini, train_file, True)
logger.debug(f'The length of the vector is {len(mini.vocab.keys())}')

mini = gen_mini_embedding(embed, mini, test_file, True)
logger.debug(f'The length of the vector is {len(mini.vocab.keys())}')

mini = gen_mini_embedding(embed, mini, train_file, False)
logger.debug(f'The length of the vector is {len(mini.vocab.keys())}')

mini = gen_mini_embedding(embed, mini, test_file, False)
logger.debug(f'The length of the vector is {len(mini.vocab.keys())}')

fname = "./output/mini.kv"
mini.save(fname)
