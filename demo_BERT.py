import torch
import numpy as np
from transformers import BertModel, BertTokenizer
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



train, test, _, _, _ = read_1(DATA_PATH)

title = train['title'].values.tolist()


input = title[:BATCH_SIZE]

#print('First 5 of input title', input[:5])

model_name = '../shopee_product_matching/BERT_Pretrained/'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

input = tokenizer(input, padding='max_length', max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
input_ids = input["input_ids"]
#print('Encoded input', input_ids[:5])


# cosine similiarity
with torch.no_grad():
    output = model(input_ids)["last_hidden_state"] #or "pooler_output"
    output = output.view(BATCH_SIZE,-1)
    similarity = []
    for i in range(BATCH_SIZE):
        similarity.append(torch.cosine_similarity(output[0,:], output[i,:], dim=0))
    similarity = torch.tensor(similarity)
    top_index = np.argsort(-similarity)[0:10]
    print('similarity', similarity, top_index)
