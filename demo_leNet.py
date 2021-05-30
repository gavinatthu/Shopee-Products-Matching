import torch
import numpy as np
from data_loader import *
from Methods import *
from evaluate import *
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = '../shopee_product_matching/'
BATCH_SIZE = 256
IMG_SIZE = 512

# train = read_sort(DATA_PATH)
train, test, train_path, test_path, labelMAP = read_1(DATA_PATH)

# 读取划分好的训练集
trainDataset = Dataloader1(train, labelMAP, train_path, IMG_SIZE, IMG_SIZE) # 实际训练用这句

# torch读取
trainLoader = torch.utils.data.DataLoader(
    trainDataset,
    BATCH_SIZE, shuffle=False, num_workers=2)

# 读取划分好的测试集
testDataset = Dataloader1(test, labelMAP, test_path, IMG_SIZE, IMG_SIZE) # 实际训练用这句

# torch读取
testLoader = torch.utils.data.DataLoader(
    testDataset,
    BATCH_SIZE, shuffle=False, num_workers=2)

# # Display image
# train_features, train_labels = next(iter(trainLoader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0,1].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.savefig("sample.png")
# print(f"Label: {label}")

# for X, y in trainLoader:
#     print("Shape of X [N, C, H, W]: ", X.shape)
#     print("Shape of y: ", y.shape, y.dtype)
#     break


#imgmodel = ShopeeImageEmbeddingNet().to(device) # resnet 50
imgmodel = NeuralNetwork().to(device) #NN 未训练
# 训练
print(imgmodel)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(imgmodel.parameters(), lr=0.001, momentum=0.9)

epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    trainLoop(trainLoader, imgmodel, loss_fn, optimizer, device)
    testLoop(testLoader, imgmodel, loss_fn, device)
print("Done!")

trainFeat = []
trainLabel = []
with torch.no_grad():
    for i, (data, y) in enumerate(trainLoader):
        data = data.to(device)
        trainLabel.append(torch.max(y.to(device), 1)[1].data.cpu().numpy()) # 实际的label
        feat = imgmodel(data) # 输出的feature
        trainFeat.append(feat.data.cpu().numpy())
trainFeat = np.squeeze(trainFeat)                         # 压缩之前多的单元素维度
trainLabel = np.squeeze(trainLabel)  
trainFeat = normalize(trainFeat)                          # 用sklearn实现的归一化 可以用numpy改写

testFeat = []
testLabel = []
with torch.no_grad():
    for i, (data, y) in enumerate(testLoader):
        data = data.to(device)
        testLabel.append(torch.max(y.to(device), 1)[1].data.cpu().numpy()) # 实际的label
        feat = imgmodel(data) # 输出的feature
        testFeat.append(feat.data.cpu().numpy())
testFeat = np.squeeze(testFeat)                         # 压缩之前多的单元素维度
testLabel = np.squeeze(testLabel) 
testFeat = normalize(testFeat)                          # 用sklearn实现的归一化 可以用numpy改写

print(modelEvaluation(testFeat, testLabel, trainFeat, trainLabel, 'F1', 1))
print(modelEvaluation(testFeat, testLabel, trainFeat, trainLabel, 'F1', 5))
print(modelEvaluation(testFeat, testLabel, trainFeat, trainLabel, 'F1', 20))
print(modelEvaluation(testFeat, testLabel, trainFeat, trainLabel, 'mAP'))
