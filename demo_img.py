import torch
import numpy as np
from data_loader import *
from Methods import *


from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
import os

os.environ["CUDA_VISIBLE_deviceS"] = "7"

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

DATA_PATH = '../shopee_product_matching/'
BATCH_SIZE = 100
IMG_SIZE = 512

#train = read_sort(DATA_PATH)
train, test, train_path, test_path = read(DATA_PATH)

imagedataset = Dataloader(train_path[:BATCH_SIZE], IMG_SIZE, IMG_SIZE)

imageloader = torch.utils.data.DataLoader(
    imagedataset,
    BATCH_SIZE, shuffle=False, num_workers=2)


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
print("Target group:",train.head())

# 简单算一个平均F1 score，由于训练和测试在同一个图片集，故取第二相似的图
y_true = np.zeros(BATCH_SIZE)
y_pred = np.zeros(BATCH_SIZE)
for INDEX in range(BATCH_SIZE):
    cor = np.dot(imagefeat[INDEX], imagefeat.T)
    top_index = np.argsort(-cor)[0:TOP_K]
    y_true[INDEX] = train.iloc[INDEX][4]
    y_pred[INDEX] = train.iloc[top_index[1]][4]
print('f1_score:',f1_score(y_true, y_pred, average='micro'))
