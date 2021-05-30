import torch
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from data_loader import *
from Methods import *

from sklearn.metrics import f1_score
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = '../shopee_product_matching/'
BATCH_SIZE = 100
IMG_SIZE = 512

start = time.time()

train, test, train_path, test_path, _ = read_1(DATA_PATH)

imagedataset = Dataloader(train, train_path, IMG_SIZE, IMG_SIZE)
test_set = Dataloader(test, test_path, IMG_SIZE, IMG_SIZE)

imageloader = torch.utils.data.DataLoader(imagedataset, BATCH_SIZE, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(test_set, BATCH_SIZE, shuffle=False, num_workers=2)


imgmodel = P_Efnetb5().to(device)
#imgmodel = P_Resnetb5().to(device)

# 训练集特征提取
imagefeat, imagelabel = [], []
with torch.no_grad():
    for data, label in tqdm(imageloader):
        data = data.to(device)
        label = label.to(device)
        imagelabel.append(label)
        feat = imgmodel(data)
        imagefeat.append(feat)
imagefeat = torch.cat(imagefeat, dim=0)
imagelabel = torch.cat(imagelabel, dim=0)
# imagefeat.shape = (31250,300)
# 通过修改最后一个隐层节点数统一到(31250,300)，和文本做特征组合，跨模态学习
# Feature -> imagefeat

# 测试集特征提取
testfeat, testlabel = [], []
with torch.no_grad():
    for data, label in tqdm(testloader):
        data = data.to(device)
        label = label.to(device)
        testlabel.append(label)
        feat = imgmodel(data)   
        testfeat.append(feat)
testfeat = torch.cat(testfeat, dim=0)
testlabel = torch.cat(testlabel, dim=0)

# testfeat.shape = (3000,300)

# Covariance Matrix
cor = torch.mm(testfeat, imagefeat.T)
# cor.shape = (3000, 31250)
# 
# 这里以下转入CPU操作

imagelabel = imagelabel.data.cpu().numpy()
testlabel = testlabel.data.cpu().numpy()
cor = cor.data.cpu().numpy()
top_index = np.argsort(-cor, axis=1)


top20_acc, top5_acc, top1_acc = 0, 0, 0
for k in range(testfeat.shape[0]):
    if (testlabel[k] in imagelabel[top_index[k]][:1]): 
        print(imagelabel[top_index[0]])
        print('Success top1! Label=', testlabel[k])
        top1_acc += 1
        top5_acc += 1
        top20_acc += 1
    elif (testlabel[k] in imagelabel[top_index[k]][:5]): 
        print('Success top5! Label=', testlabel[k])
        top5_acc += 1
        top20_acc += 1
    elif (testlabel[k] in imagelabel[top_index[k]][:20]): 
        print('Success top20! Label=', testlabel[k])
        top20_acc += 1
    else:
        print('Fault Label=', testlabel[k])
print('acc1=', top1_acc)
print('acc5=', top5_acc)
print('acc20=', top20_acc)
print('time=', time.time() - start)

