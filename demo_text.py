import torch
import numpy as np
from data_loader import *
from Methods import *
import os
import time
import gensim
from gensim.test.utils import datapath, get_tmpfile

os.environ["CUDA_VISIBLE_deviceS"] = "7"
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


DATA_PATH = '../shopee_product_matching/'

BATCH_SIZE = 50
MAX_LENGTH = 24

if os.path.exists(DATA_PATH + 'train_set.csv'):
    print('Dataset exists!')
else:
    _, _, _, _ = read_1(DATA_PATH)     # 事实上这一步作用只是生成train_set.csv和test_set.csv，pd.dataframe到torchtext的接口没有写，word2vec函数还要再读一遍

# 加载分词后的语料库
TEXT, LABEL, train, test = word2id(DATA_PATH, BATCH_SIZE, MAX_LENGTH)


# 直接存list的方案
sentences, train_sentences, train_label, test_sentences, test_label = sen2list(train, test)


MODEL_PATH = '/data1/shopee/shopee.model'                  # Modelpath需要足够大的空间存储，服务器分盘下需要放到data盘里（8Gb左右）


if os.path.exists(MODEL_PATH):
    print('Loading model...')
    model = gensim.models.FastText.load(MODEL_PATH)

else:
    print('Generating model...')
    model = gensim.models.fasttext.load_facebook_model(gensim.test.utils.datapath('/home/common/ljw/shopee/crawl-300d-2M-subword.bin')) #这里需要用绝对路径
    model.save(MODEL_PATH)
    #model = gensim.models.FastText(sentences, vector_size=100, window=20, min_count=0)         #这个是只用本地title训练的FastText
    #model = gensim.models.Word2Vec(sentences, vector_size=100, window=20, min_count=0)         #这个是只用本地title训练的word2vec
    model.build_vocab(train_sentences, update=True)
    model.train(train_sentences, total_examples=len(train_sentences), epochs=model.epochs)


print(model)


start = time.time()
k = 0
top20_acc, top5_acc, top1_acc = 0, 0, 0
for k in range(len(test_sentences)):
    sen1, label1 = test_sentences[k], test_label[k]
    print(sen1, label1)

    sim_total = []

    # 这一步循环非常慢 事实上已经构建了trainsentences的list，可以直接计算sentence和list of sentence的余弦距离
    for i in range(len(train_sentences)):
        sim = model.wv.n_similarity(sen1,train_sentences[i])
        sim_total.append(sim)

    sim_total = np.array(sim_total)
    sim_index = np.argsort(-sim_total)

    if (test_label[k] in [train_label[i] for i in sim_index[:1]]): 
        print('Success top1! Label=', test_label[k])
        top1_acc += 1
        top5_acc += 1
        top20_acc += 1
    elif (test_label[k] in [train_label[i] for i in sim_index[:5]]): 
        print('Success top5! Label=', test_label[k])
        top5_acc += 1
        top20_acc += 1
    elif (test_label[k] in [train_label[i] for i in sim_index[:20]]): 
        print('Success top20! Label=', test_label[k])
        top20_acc += 1
    else:
        print('Fault Label=', test_label[k])


print('acc1=', top1_acc)
print('acc5=', top5_acc)
print('acc20=', top20_acc)
print('time=', time.time() - start)



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
