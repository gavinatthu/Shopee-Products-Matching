import torch
import numpy as np
from data_loader import *
from Methods import *
import os

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

os.environ["CUDA_VISIBLE_deviceS"] = "7"
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


DATA_PATH = '../shopee_product_matching/'

BATCH_SIZE = 50
MAX_LENGTH = 10

train, test, _, _ = read_1(DATA_PATH)     # 事实上这一步作用只是生成train_set.csv和test_set.csv，pd.dataframe到torchtext的接口没有写，word2vec函数还要再读一遍

TEXT, LABEL, train_iter, test_iter = word2id(DATA_PATH, BATCH_SIZE, MAX_LENGTH)


test_batch = next(iter(test_iter))
test_input, test_label = test_batch.title, test_batch.label

print('test_label', test_batch.dataset.examples[0].title, test_batch.dataset.examples[0].label)

if os.path.exists('w2v.model'):
    print(1)
    model = Word2Vec.load("w2v.model")
else:
    model = KeyedVectors.load_word2vec_format(DATA_PATH + 'glove.6B.300d.txt', binary=False, no_header=True)
    model.save_word2vec_format(fname='w2v.model')

print(model)


train_sentences, test_sentences = [], []
with torch.no_grad():
    for idx, batch in enumerate(train_iter):
        text, label = batch.title, batch.label
        # 构建超大list: sentences
        for i in range(BATCH_SIZE):
            train_sentences.append(batch.dataset.examples[i].title)

    for idx, batch in enumerate(test_iter):
        for i in range(BATCH_SIZE):
            test_sentences.append(batch.dataset.examples[i].title)

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
#利用torch搭建单层embedding网络测试，不work


model = LSTM(TEXT.vocab.vectors)
model.train()


test_output = model(test_input)
test_item = test_output[0,:]
test_item = test_item.repeat(BATCH_SIZE,1)
max_total = 0.0

with torch.no_grad():
    for idx, batch in enumerate(train_iter):
        text, label = batch.title, batch.label
        
        predicted = model(text)

        similarity = torch.cosine_similarity(test_item, predicted, dim=1)

        max_sim = torch.max(similarity)
        
        if max_sim > max_total:
            print(max_sim)
            max_total = max_sim
            #id = idx*BATCH_SIZE + similarity.argmax()
            print('target_label', batch.dataset.examples[similarity.argmax()].title, batch.dataset.examples[similarity.argmax()].label)
'''      



'''
vocab = TEXT.vocab

print(vocab.vectors[vocab['you','bag']])

batch = next(iter(train_iter))
input = batch.title

model = RNN(TEXT.vocab.vectors)
output = model(input)

print(output)

output = id2vec(TEXT.vocab.vectors, input).permute(1,0,2) # 输出Batch_size*Max_length*300 的训练集矩阵，将其和测试集比对

print(input.shape)


test_batch = next(iter(test_iter))
test_input = test_batch.title
test_output = id2vec(TEXT.vocab.vectors, test_input).permute(1,0,2)

#x = test_output[:,0,:]
#print(torch.cosine_similarity(x, output[:,0,:], dim=0))
'''
'''
with torch.no_grad():
    similarity = []
    for i in range(BATCH_SIZE):
        similarity.append(torch.cosine_similarity(x, output[:,i,:], dim=1))
    print(similarity)
    similarity = torch.tensor(similarity)
    top_index = torch.argsort(-similarity)[0:10]
    print('similarity', similarity, top_index)

    #print('Test title: ', test_iter.dataset.examples[0].title)
    #print('Test ids: ', test_item)
'''




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
