import numpy as np
import gensim
from code_felix.core.config import *
import pandas as pd
from code_felix.core.feature import *
from tqdm import tqdm

import jieba

@timed()
def filter_duplicate_words(file_list):
    jieba.load_userdict(jieba_dict)

    logger.debug(f'load jieba dict:{len(list(jieba.dt.FREQ.keys()))} from file:{jieba_dict}')
    word_count = 0
    word_set = set()
    for cut_all in [True, False]:
        for file in file_list:
            input_text = pd.read_csv(file, encoding='gb18030', delimiter='\t', header=None)
            input_text['jieba'] = input_text.iloc[:, 2].apply(lambda text: ' '.join(jieba.cut(str(text), cut_all)))
            for index, text in enumerate(input_text['jieba'].values):
                for word in text.strip().split(' '):
                    word_set.add(word)
                    word_count += 1
            logger.debug(f'There are {len(word_set)} words after file:{file}')
    logger.debug(f'There are {len(word_set)} word were parser from word_count:{word_count} file_list:{file_list}')
    return sorted(list(word_set))


@timed()
def gen_mini_embedding(wv_from_text, word_list):
    from multiprocessing.dummy import Pool

    from functools import partial

    partition_num = 8
    import math
    partition_length = math.ceil(len(word_list)/partition_num)

    partition_list = [ word_list[i:i+partition_length]  for i in range(0, len(word_list), partition_length )]
    logger.debug(f'The word list split to {len(partition_list)} partitions:{[ len(partition) for partition in partition_list]}')
    thread_pool = Pool(processes=partition_num)
    process = partial(gen_mini_partition,wv_from_text=wv_from_text )

    wv_list = thread_pool.map(process, partition_list)
    thread_pool.close(); thread_pool.join()

    del wv_from_text

    mini = merge_Word2Vec(wv_list)

    return mini


@timed()
def gen_mini_partition( word_set,  wv_from_text):
    if local:
        word_set = word_set[:3000]
        logger.debug("Run app with local model")

    mini = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size)
    #for i in tqdm(range(len(word_set)), desc=f'Queue:{id(word_set)}'):
    for i in range(len(word_set)):
        word = word_set[i]
        if word in wv_from_text and word not in mini:
            mini[word] = wv_from_text[word]
        elif word not in wv_from_text and len(word) == 1:
            logger.debug(f'Canot find vec for:1,{word}')
            mini[word] = np.zeros(vector_size)
        elif word not in wv_from_text and len(word) > 1:
            vector = wordVec(word, wv_from_text, 1, 3)
            if vector is not None:
                mini[word] = vector
            else:
                logger.debug(f'Canot find vec for:{len(word)},{word}')
                mini[word] = np.zeros(vector_size)
        else:
            mini[word] = np.zeros(vector_size)
    return mini

@timed()
def merge_Word2Vec(vec_list):
    for sn, mini_vec in enumerate(vec_list):
        partition_file = f"./output/mini_p{sn}.kv"
        mini_vec.save(partition_file)

    mini_all = None
    for mini_vec in vec_list:
        if mini_all is None:
            mini_all = mini_vec
        else:
            for entry in mini_vec.vocab.keys():
                mini_all[entry] = mini_vec[entry]
    logger.debug(f'The length of the merge vector is {len(mini_all.vocab.keys())}')
    return mini_all




if __name__ == '__main__':

    embed = load_embedding(word2vec_model)

    word_list = filter_duplicate_words([train_file, test_file])

    mini = gen_mini_embedding(embed, word_list)
    logger.debug(f'The length of the vector is {len(mini.vocab.keys())}')

    fname = "./output/mini_merge.kv"
    mini.save(fname)

    #
    # mini = gen_mini_embedding(embed, mini, test_file, True)
    # logger.debug(f'The length of the vector is {len(mini.vocab.keys())}')
    # fname = "./output/mini.kv"
    # mini.save(fname)
    #
    # mini = gen_mini_embedding(embed, mini, train_file, False)
    # logger.debug(f'The length of the vector is {len(mini.vocab.keys())}')
    #
    # mini = gen_mini_embedding(embed, mini, test_file, False)
    # logger.debug(f'The length of the vector is {len(mini.vocab.keys())}')
    #
    # #fname = "./output/mini.kv"
    # mini.save(fname)
