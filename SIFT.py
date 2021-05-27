import cv2 as cv
import numpy as np
import time
from tqdm import tqdm
from data_loader1 import *
from Methods import *
import os



DATA_PATH = '../shopee_product_matching/'

start = time.time()
train_path, test_path, train_label, test_label = read_only(DATA_PATH)

'''
if os.path.exists("./logs/train_feat.npy"): # 读取特征矩阵
    train_feat = np.load("./logs/train_feat.npy")
    test_feat = np.load("./logs/test_feat.npy")
    
else: # 特征提取
'''
train_feat, test_feat = [], []
for train_path_tmp in tqdm(test_path):
    img_tmp = cv.imread(train_path_tmp,cv.IMREAD_GRAYSCALE)
    orb = cv.ORB_create()
    _, feat = orb.detectAndCompute(img_tmp,None)
    '''
    if len(feat)<=300:
        # 有的图片输出特征点不到300个，这里采用padding的方法
        feat = np.pad(feat,((300-len(feat),0),(0,0)),'constant', constant_values=0)
    '''
    train_feat.append(feat[:300])
    

for test_path_tmp in tqdm(test_path):
    img_tmp = cv.imread(test_path_tmp,cv.IMREAD_GRAYSCALE)
    orb = cv.ORB_create()
    _, feat = orb.detectAndCompute(img_tmp,None)
    '''
    if len(feat)<=300:
        # 有的图片输出特征点不到300个，这里采用padding的方法
        feat = np.pad(feat,((300-len(feat),0),(0,0)),'constant', constant_values=0)
    '''
    test_feat.append(feat[:300])
    

train_feat = np.array(train_feat) # 特征矩阵： (31250, 300, 32)
np.save("./logs/train_feat.npy",train_feat)

test_feat = np.array(test_feat)   # (3000, 300, 32)
np.save("./logs/test_feat.npy",test_feat)


# create BFMatcher object
#bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
sim = np.zeros((len(test_feat),len(train_feat)))

for k, test in enumerate(tqdm(test_feat[:2])):
    for i in range(len(train_feat)):

        # 计算相似点，len(matches)表示相似度
        matches = bf.match(test,train_feat[i])

        sim[k][i] = len(matches)

top_k = 5
top_k_idx=np.argsort(-sim, axis=1)[:,:5]
print(top_k_idx.shape)


print(sim[0][top_k_idx[0]])
print(top_k_idx[0])
print(test_label[0], train_label[top_k_idx[0]])
