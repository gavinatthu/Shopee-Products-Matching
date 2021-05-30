import torch
import numpy as np
from data_loader import *
from Methods import *
from evaluate import *
import os
from tqdm import tqdm
from sklearn.preprocessing import normalize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")

DATA_PATH = '../shopee_product_matching/'

BATCH_SIZE = 128
MAX_LENGTH = 24

if os.path.exists(DATA_PATH + 'train_set.csv'):
    print('Dataset exists!')
else:
    _, _, _, _ = read_1(DATA_PATH)     # 事实上这一步作用只是生成train_set.csv和test_set.csv，pd.dataframe到torchtext的接口没有写，word2vec函数还要再读一遍

itrain, itest, itrain_path, itest_path, labelMAP = read_1(DATA_PATH)

# 加载分词后的语料库
_, _, ttrain, ttest = word2id(DATA_PATH, BATCH_SIZE, MAX_LENGTH)

# 直接存list的方案
sentences, train_sentences, train_label, test_sentences, test_label = sen2list(ttrain, ttest)

MODEL_PATH = '/data1/shopee/shopee.model'                  # Modelpath需要足够大的空间存储，服务器分盘下需要放到data盘里（8Gb左右）

model1 = TF_IDF(train_sentences)

# 用TF-IDF生成文字的特征表
trainFeatt = []
testFeatt = []

for train_words in tqdm(train_sentences):
    feat = model1.Feat_ext(train_words)
    trainFeatt.append(feat)

trainLabelt = [label2SN(labelMAP, int(lb)) for lb in train_label]

for test_words in tqdm(test_sentences):
    feat = model1.Feat_ext(test_words)
    testFeatt.append(feat)

testLabelt = [label2SN(labelMAP, int(lb)) for lb in test_label]

IMG_SIZE = 256

trainDataset = DataloaderImgText(trainFeatt, itrain, labelMAP, itrain_path, IMG_SIZE, IMG_SIZE)
trainLoader = torch.utils.data.DataLoader(
    trainDataset,
    BATCH_SIZE, shuffle=False, num_workers=2)

testDataset = DataloaderImgText(testFeatt, itest, labelMAP, itest_path, IMG_SIZE, IMG_SIZE)
testLoader = torch.utils.data.DataLoader(
    testDataset,
    BATCH_SIZE, shuffle=False, num_workers=2)

imgmodel = embeddingNetwork() #NN 未训练
# 训练
print(imgmodel)
if os.path.exists('embedding_model.pt'):
    print('Model exists!')
    imgmodel.load_state_dict(torch.load('embedding_model.pt'))
    imgmodel.to(device)
else:
    print('Start training')
    imgmodel.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(imgmodel.parameters(), lr=0.001, momentum=0.9)

    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        trainLoopti(trainLoader, imgmodel, loss_fn, optimizer, device)
        testLoopti(testLoader, imgmodel, loss_fn, device)
    print("Done!")

trainFeatti = np.empty([0,11014])
trainLabelti = np.empty([0],dtype = np.int)
with torch.no_grad():
    for i, (text, img, y) in enumerate(trainLoader):
        text, img= text.to(device), img.to(device)
        trainLabelti = np.concatenate((trainLabelti, torch.max(y.to(device), 1)[1].data.cpu().numpy())) # 实际的label
        feat = imgmodel(text.float(), img) # 输出的feature
        trainFeatti = np.concatenate((trainFeatti, feat.data.cpu().numpy()), axis = 0)
trainLabelti = trainLabelti.tolist()

testFeatti = np.empty([0,11014])
testLabelti = np.empty([0],dtype = np.int)
with torch.no_grad():
    for i, (text, img, y) in enumerate(testLoader):
        text, img= text.to(device), img.to(device)
        testLabelti = np.concatenate((testLabelti, torch.max(y.to(device), 1)[1].data.cpu().numpy())) # 实际的label
        feat = imgmodel(text.float(), img) # 输出的feature
        testFeatti = np.concatenate((testFeatti, feat.data.cpu().numpy()), axis = 0)
testLabelti = testLabelti.tolist()

print(modelEvaluation(testFeatti, testLabelti, trainFeatti, trainLabelti, 'F1', 1))
print(modelEvaluation(testFeatti, testLabelti, trainFeatti, trainLabelti, 'F1', 5))
print(modelEvaluation(testFeatti, testLabelti, trainFeatti, trainLabelti, 'F1', 20))
# print(modelEvaluation(testFeatti, testLabelti, trainFeatti, trainLabelti, 'AUC')) # 类别太多，太慢
print(modelEvaluation(testFeatti, testLabelti, trainFeatti, trainLabelti, 'mAP'))
