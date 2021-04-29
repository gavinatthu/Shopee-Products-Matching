import torch
import numpy as np
from data_loader import *
from transformers import BertModel, BertTokenizer
import os
os.environ["CUDA_VISIBLE_deviceS"] = "7"
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


DATA_PATH = '../shopee_product_matching/'
BATCH_SIZE = 1000
MAX_LENGTH = 36

'''
TEXT, train = textLoader(DATA_PATH, BATCH_SIZE)
'''

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
# 2-Norm
with torch.no_grad():
    output = model(input_ids)["last_hidden_state"] #or "pooler_output"
    print('Output shape = Batch*Length*Dim:', output.shape)
    similarity = []
    
    for i in range(BATCH_SIZE):
        similarity.append(torch.norm(output[0,:,:] - output[i,:,:]))
    similarity = torch.tensor(similarity)
    top_index = np.argsort(similarity)[0:10]
    print('similarity', similarity, top_index)
'''
