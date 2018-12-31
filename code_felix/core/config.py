word2vec_model = './input/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt'
train_file = './input/data_train.csv'
test_file = './input/data_test.csv'
jieba_dict = './input/jieba.txt'
vector_size=200
from file_cache.utils.util_log import *
try:
    from code_felix.core.config_local import *
except Exception as e:
    logger.exception(e)
    logger.debug("There is no local config")

