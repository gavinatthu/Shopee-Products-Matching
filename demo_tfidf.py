import torch
import numpy as np
from data_loader import *
from Methods import *
import os
import time
import gensim

print(time.time())

os.environ["CUDA_VISIBLE_deviceS"] = "7"
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


DATA_PATH = '../shopee_product_matching/'

BATCH_SIZE = 50
MAX_LENGTH = 24

if os.path.exists(DATA_PATH + 'train_set.csv'):
    print('Dataset exists!')
else:
    _, _, _, _ = read_1(DATA_PATH)     # 事实上这一步作用只是生成train_set.csv和test_set.csv，pd.dataframe到torchtext的接口没有写，word2vec函数还要再读一遍

TEXT, LABEL, train, test = word2id(DATA_PATH, BATCH_SIZE, MAX_LENGTH)
sentences, train_sentences, train_label, test_sentences, test_label = sen2list(train, test)


# 建立字典
dictionary = gensim.corpora.Dictionary(train_sentences)

# 分词corpus
corpus = [dictionary.doc2bow(item) for item in train_sentences]

# Tfidf算法 计算tf值
tf = gensim.models.TfidfModel(corpus)

# 通过token2id得到特征数（字典里面的键的个数）
num_features = len(dictionary.token2id.keys())

# 计算稀疏矩阵相似度，建立一个索引
index = gensim.similarities.MatrixSimilarity(tf[corpus], num_features=num_features)




top20_acc, top5_acc, top1_acc = 0, 0, 0
start = time.time()
for k in range(len(test_sentences)):

    test_words, target = test_sentences[k], test_label[k]
    #print(test_words, label1)
    
    new_vec = dictionary.doc2bow(test_words)

    sim_total = index[tf[new_vec]]
    sim_total = np.array(sim_total)

    sim_index = np.argsort(-sim_total)

    if (target in [train_label[i] for i in sim_index[:1]]): 
        print('Success top1! Label=', target)
        top1_acc += 1
        top5_acc += 1
        top20_acc += 1
    elif (target in [train_label[i] for i in sim_index[:5]]): 
        print('Success top5! Label=', target)
        top5_acc += 1
        top20_acc += 1
    elif (target in [train_label[i] for i in sim_index[:20]]): 
        print('Success top10! Label=', target)
        top20_acc += 1
    else:
        print('Fault Label=', target)

print('acc1=', top1_acc)
print('acc5=', top5_acc)
print('acc10=', top20_acc)
print('time=', time.time() - start)
