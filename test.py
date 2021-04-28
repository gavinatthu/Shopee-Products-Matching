import torch
import numpy as np
from data_loader import read, Dataloader
from Methods import *


from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score

import os

os.environ["CUDA_VISIBLE_deviceS"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 改为相对路径，数据在上层文件夹的shopee-product-matching中
DATA_PATH = '..\\shopee-product-matching\\'
BATCH_SIZE = 48 # 爆了我的2066，不设置成全部训练集如何防止训练集中没有某一类的点，导致无法检测算法的准确性？
IMG_SIZE = 512

#train = read_sort(DATA_PATH)
train, test, train_path, test_path = read(DATA_PATH)

imagedataset = Dataloader(train_path[:BATCH_SIZE], IMG_SIZE, IMG_SIZE)

imageloader = torch.utils.data.DataLoader(
    imagedataset,
    BATCH_SIZE, shuffle=False, num_workers=0) # num_workers=2我在自己机子跑不起来


imgmodel = ShopeeImageEmbeddingNet().to(device)

imagefeat = []
with torch.no_grad():
    for i, data in enumerate(imageloader):
        data = data.to(device)
        feat = imgmodel(data)
        imagefeat.append(feat.data.cpu().numpy())


imagefeat = np.squeeze(imagefeat)                         # 压缩之前多的单元素维度
imagefeat = normalize(imagefeat)                          # 用sklearn实现的归一化 可以用numpy改写


INDEX = 2       # 输入待检索的图片序号
TOP_K = 5       # 输出前5个最相似的序号

cor = np.dot(imagefeat[INDEX], imagefeat.T)
top_index = np.argsort(-cor)[0:TOP_K]                     # 对于后验可以划分一个合适的界限

print("Posterior:",cor[top_index])
print("Predicted group:",train.iloc[top_index])
print("Target group:",train.iloc[INDEX])

# 简单算一个平均F1 score，由于训练和测试在同一个图片集，故取第二相似的图
y_true = np.zeros(BATCH_SIZE)
y_pred = np.zeros(BATCH_SIZE)
for INDEX in range(BATCH_SIZE):
    cor = np.dot(imagefeat[INDEX], imagefeat.T)
    top_index = np.argsort(-cor)[0:TOP_K]
    y_true[INDEX] = train.iloc[INDEX][4]
    y_pred[INDEX] = train.iloc[top_index[1]][4]
print(f1_score(y_true, y_pred, average='micro'))