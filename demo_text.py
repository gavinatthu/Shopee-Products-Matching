import torch
import numpy as np
from data_loader import *
from Methods import*
from evaluate import calAccuracy, conv2Top1, calF1score
import os
import time
from tqdm import tqdm
import sklearn

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda")


DATA_PATH = '../shopee_product_matching/'

BATCH_SIZE = 50
MAX_LENGTH = 24

if os.path.exists(DATA_PATH + 'train_set.csv'):
    print('Dataset exists!')
else:
    _, _, _, _ = read_1(DATA_PATH)     # 事实上这一步作用只是生成train_set.csv和test_set.csv，pd.dataframe到torchtext的接口没有写，word2vec函数还要再读一遍

# 加载分词后的语料库
_, _, train, test = word2id(DATA_PATH, BATCH_SIZE, MAX_LENGTH)

# 直接存list的方案
sentences, train_sentences, train_label, test_sentences, test_label = sen2list(train, test)

MODEL_PATH = '/data1/shopee/shopee.model'                  # Modelpath需要足够大的空间存储，服务器分盘下需要放到data盘里（8Gb左右）


model1 = TF_IDF(train_sentences)
model2 = Fast_Text(train_sentences, MODEL_PATH)


start = time.time()

# 存储预测label的列表
pred_label = []
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

print(feat_total1.shape, feat_total2.shape)
'''
feat_total = np.array(feat_total)[:,:,1]

pred_label = np.array(pred_label)


top1_acc = calAccuracy(test_label, pred_label, topN=1)
top5_acc = calAccuracy(test_label, pred_label, topN=5)
top20_acc = calAccuracy(test_label, pred_label, topN=20)
f1score = calF1score(test_label, pred_label[:20], topN=1, average='micro')

print('acc1=', f1score)
print('acc1=', top1_acc)
print('acc5=', top5_acc)
print('acc20=', top20_acc)
print('time=', time.time() - start)
'''

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




#利用torch搭建单层word2vec embedding网络测试，不work
'''
model = W2V(TEXT.vocab.vectors)
model.train()

test_output = model(test_input)
test_item = test_output[1,:]
test_item = test_item.repeat(BATCH_SIZE, 1)

print('test_label', test_batch.dataset.examples[1].title, test_batch.dataset.examples[1].label)

max_total = -1.0
max_list = []
with torch.no_grad():
    for idx, batch in enumerate(train_iter):
        text, label = batch.title, batch.label
        predicted = model(text)
        similarity = torch.cosine_similarity(test_item, predicted, dim=1)
        
        max_sim = torch.max(similarity)
        
        if(max_sim > max_total):
            id_max = similarity.argmax()
            print(max_sim,similarity,id_max)
            max_total = max_sim
            max_list.append(max_total)
            #id = idx*BATCH_SIZE + similarity.argmax()
            print('target_label', batch.dataset.examples[id_max].title, batch.dataset.examples[id_max].label)
        else:
            pass

    print(max_list)
'''




# 利用预训练的BERT做的特征提取，不work
'''
from transformers import BertModel, BertTokenizer

train, test, _, _ = read(DATA_PATH)
title = train['title'].values.tolist()


input = title[:BATCH_SIZE]

print('First 5 of input title', input[:5])

model_name = '../shopee_product_matching/BERT_Pretrained/'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

input = tokenizer(input, padding='max_length', max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
input_ids = input["input_ids"]
print('Encoded input', input_ids[:5])


# cosine similiarity
with torch.no_grad():
    output = model(input_ids)["last_hidden_state"] #or "pooler_output"
    print('Output shape = Batch*Length*Dim:', output.shape)
    output = output.view(BATCH_SIZE,-1)
    print('Output shape = Batch*Length*Dim:', output.shape)
    similarity = []
    for i in range(BATCH_SIZE):
        similarity.append(torch.cosine_similarity(output[0,:], output[i,:], dim=0))
    similarity = torch.tensor(similarity)
    top_index = np.argsort(-similarity)[0:10]
    print('similarity', similarity, top_index)
'''
