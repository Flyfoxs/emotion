from tqdm import tqdm
from file_cache.utils.util_date import timed

word2vec_model = './input/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt'


@timed()
def gendict(inputFile, ouputFile):
    output_f = open(ouputFile, 'a', encoding='utf8')
    with open(inputFile, "r", encoding='ISO-8859-1') as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        for i in tqdm(range(vocab_size)):
            line = f.readline()
            lists = line.split(' ')
            word = lists[0]
            try:
                word = word.encode('ISO-8859-1').decode('utf8')
                output_f.write(word + '\n')
            except:
                pass
        output_f.close()
        f.close()


gendict(word2vec_model, './input/jieba.txt')
