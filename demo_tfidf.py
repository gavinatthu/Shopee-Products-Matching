import torch
import numpy as np
from data_loader import *
from Methods import *
import os

import gensim
#from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

os.environ["CUDA_VISIBLE_deviceS"] = "7"
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


DATA_PATH = '../shopee_product_matching/'

BATCH_SIZE = 50
MAX_LENGTH = 5

train, test, _, _ = read_1(DATA_PATH)     # 事实上这一步作用只是生成train_set.csv和test_set.csv，pd.dataframe到torchtext的接口没有写，word2vec函数还要再读一遍

TEXT, LABEL, train, test = word2id(DATA_PATH, BATCH_SIZE, MAX_LENGTH)

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

dictionary = gensim.corpora.Dictionary(sentences)

corpus = [dictionary.doc2bow(item) for item in sentences]


# 4.通过TF模型算法，计算出tf值
tf = gensim.models.TfidfModel(corpus)
# 5.通过token2id得到特征数（字典里面的键的个数）
num_features = len(dictionary.token2id.keys())
# 6.计算稀疏矩阵相似度，建立一个索引
index = gensim.similarities.MatrixSimilarity(tf[corpus], num_features=num_features)

# 7.处理测试数据
test_words = test_sentences[0]

print(test_words, test_label[0])

# 8.新的稀疏向量
new_vec = dictionary.doc2bow(test_words)
# 9.算出相似度
sims = index[tf[new_vec]]
sim_total = np.array(sims)
sim_total = np.array(sim_total)
sim_index = np.argsort(-sim_total)

print(sim_total[sim_index][:10])
print(sim_index[:10])

print(train_sentences[sim_index[1]], train_label[sim_index[1]])
print(train_sentences[sim_index[2]], train_label[sim_index[2]])
print(train_sentences[sim_index[3]], train_label[sim_index[3]])
