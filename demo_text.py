import torch
import numpy as np
from data_loader import *
from Methods import *
from evaluate import *
import os
import time
from tqdm import tqdm
import sklearn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")


DATA_PATH = '../shopee_product_matching/'

BATCH_SIZE = 50
MAX_LENGTH = 24

if os.path.exists(DATA_PATH + 'train_set.csv'):
    print('Dataset exists!')
else:
    _, _, _, _, _ = read_1(DATA_PATH)     # 事实上这一步作用只是生成train_set.csv和test_set.csv，pd.dataframe到torchtext的接口没有写，word2vec函数还要再读一遍

# 加载分词后的语料库
_, _, train, test = word2id(DATA_PATH, BATCH_SIZE, MAX_LENGTH)


# 直接存list的方案
sentences, train_sentences, train_label, test_sentences, test_label = sen2list(train, test)

MODEL_PATH = '/data1/shopee/shopee.model'                  # Modelpath需要足够大的空间存储，服务器分盘下需要放到data盘里（8Gb左右）


model1 = TF_IDF(train_sentences)
model2 = Fast_Text(train_sentences, MODEL_PATH)


start = time.time()


train_feat1, train_feat2 = [], []

for train_words in tqdm(train_sentences):

    feat1 = model1.Feat_ext(train_words)
    feat2 = model2.Feat_ext(train_words)
    #sims = np.array(sims)

    train_feat1.append(feat1)
    train_feat2.append(feat2)
    # 按照后验概率从大到小排序
    #sim_index = np.argsort(-sims)
    #pred_label.append([train_label[i] for i in sim_index[:20]])

# 存储预测label的列表
train_feat1 = sklearn.preprocessing.normalize(train_feat1, axis=1)
train_feat2 = sklearn.preprocessing.normalize(train_feat2, axis=1)


feat_total1, feat_total2 = [], []

for test_words in tqdm(test_sentences):

    feat1 = model1.Feat_ext(test_words)
    feat2 = model2.Feat_ext(test_words)
    #sims = np.array(sims)

    feat_total1.append(feat1)
    feat_total2.append(feat2)
    # 按照后验概率从大到小排序
    #sim_index = np.argsort(-sims)
    #pred_label.append([train_label[i] for i in sim_index[:20]])


#feat_total1 = np.array(feat_total1)
feat_total1 = sklearn.preprocessing.normalize(feat_total1, axis=1)
feat_total2 = sklearn.preprocessing.normalize(feat_total2, axis=1)

labelMAP = buildLabelMAP(DATA_PATH)

testLabelt = [label2SN(labelMAP, int(lb)) for lb in test_label]

trainLabelt = [label2SN(labelMAP, int(lb)) for lb in train_label]


F1_1 = modelEvaluation(feat_total1, testLabelt, train_feat1, trainLabelt, 'F1')
F1_2 = modelEvaluation(feat_total2, testLabelt, train_feat2, trainLabelt, 'F1')
print(F1_1, F1_2)

F1_1 = modelEvaluation(feat_total1, testLabelt, train_feat1, trainLabelt, 'F1',5)
F1_2 = modelEvaluation(feat_total2, testLabelt, train_feat2, trainLabelt, 'F1',5)
print(F1_1, F1_2)

MAP_1 = modelEvaluation(feat_total1, testLabelt, train_feat1, trainLabelt, 'mAP')
MAP_2 = modelEvaluation(feat_total2, testLabelt, train_feat2, trainLabelt, 'mAP')
print(MAP_1, MAP_2)

AUC_1 = modelEvaluation(feat_total1, testLabelt, train_feat1, trainLabelt, 'AUC')
AUC_2 = modelEvaluation(feat_total2, testLabelt, train_feat2, trainLabelt, 'AUC')
print(AUC_1, AUC_2)




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


test_batch = next(iter(test_iter))
test_input, test_label = test_batch.title, test_batch.label

max_sim = 0
with torch.no_grad():
    for i in range(len(train_sentences)):
        
        sim = model.n_similarity(train_sentences[i],test_sentences[0])

        if sim > max_sim:
            id_max = i
            max_sim = sim



print(max_sim)
print(sentences[id_max])
'''


# 利用预训练的BERT做的特征提取，不work
