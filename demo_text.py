import torch
import numpy as np
from data_loader import *
from Methods import *
import os

import gensim
from gensim.test.utils import datapath, get_tmpfile
#from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

os.environ["CUDA_VISIBLE_deviceS"] = "7"
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


DATA_PATH = '../shopee_product_matching/'

BATCH_SIZE = 50
MAX_LENGTH = 5

train, test, _, _ = read_1(DATA_PATH)     # 事实上这一步作用只是生成train_set.csv和test_set.csv，pd.dataframe到torchtext的接口没有写，word2vec函数还要再读一遍

TEXT, LABEL, train, test = word2id(DATA_PATH, BATCH_SIZE, MAX_LENGTH)


# 加载分词后的语料库
'''
# 利用迭代器的方案
sentences, train_sentences, test_sentences = [], [], []
with torch.no_grad():
    for idx, batch in enumerate(train_iter):
        text, label = batch.title, batch.label
        # 构建超大list: sentences
        for i in range(BATCH_SIZE):
            sentences.append(batch.dataset.examples[i].title)
            train_sentences.append(batch.dataset.examples[i].title)

    for idx, batch in enumerate(test_iter):
        for i in range(BATCH_SIZE):
            sentences.append(batch.dataset.examples[i].title)
            test_sentences.append(batch.dataset.examples[i].title)
'''

# 直接存list的方案
sentences, train_sentences, train_label, test_sentences, test_label = [], [], [], [], []
with torch.no_grad():
    for i in range(len(train)):
        sentences.append(train[i].title)
        train_sentences.append(train[i].title)
        train_label.append(train[i].label)
    for i in range(len(test)):
        sentences.append(test[i].title)
        test_sentences.append(test[i].title)
        test_label.append(test[i].label)


if os.path.exists('/data1/shopee.model'):
    print('Loading model...')
    model = gensim.models.FastText.load('/data1/shopee.model')
else:

    print('Generating model...')
    #model = gensim.models.FastText(sentences, vector_size=100, window=20, min_count=0)
    #model = gensim.models.KeyedVectors.load_word2vec_format(DATA_PATH + 'wiki-news-300d-1M.vec', binary=True)
    #model = gensim.models.FastText.load_fasttext_format(DATA_PATH + 'crawl-300d-2M-subword.bin', encoding='utf-8')
    model = gensim.models.fasttext.load_facebook_model(gensim.test.utils.datapath('/home/common/ljw/shopee/crawl-300d-2M-subword.bin'))
    #model = gensim.models.Word2Vec(sentences, vector_size=100, window=20, min_count=0)
    model.save('/data1/shopee.model')

print(model)


k = 0
acc = 0
for k in range(100):
    sen1, label1 = test_sentences[k], test_label[k]
    print(sen1, label1)

    sim_total = []
    max_sim = -1

    for i in range(len(train_sentences)):
        sim = model.wv.n_similarity(sen1,train_sentences[i])
        sim_total.append(sim)

    sim_total = np.array(sim_total)
    sim_index = np.argsort(-sim_total)

    #print(sim_total[sim_index][:10])
    #print(sim_index[:10])

    if train_label[sim_index[0]]==test_label[k]:
        print('Success! Label=', test_label[k])
        acc += 1
    elif train_label[sim_index[1]]==test_label[k]:
        print('Success in 2nd! Label=', test_label[k])
    elif train_label[sim_index[2]]==test_label[k]:
        print('Success in 3nd! Label=', test_label[k])


print('acc=', acc)
