import numpy as np
import pandas as pd
import cv2, matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm_notebook
import gc

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

import os
from data_loader import Dataloader

os.environ["CUDA_VISIBLE_deviceS"] = "5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = '/data/shopee_product_matching/'
'''
train = pd.read_csv(DATA_PATH + 'train.csv')
train['image'] = DATA_PATH + 'train_images/' + train['image']
tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
train['target'] = train.label_group.map(tmp)
train = train.sort_values(by='label_group')

class ShopeeImageDataset(Dataset):
    def __init__(self, img_path, transform):
        self.img_path = img_path
        self.transform = transform
        
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.img_path)

imagedataset = ShopeeImageDataset(
    train['image'].values[:100],
    transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))
'''

imagedataset = Dataloader(DATA_PATH, 512, 512)

imageloader = torch.utils.data.DataLoader(
    imagedataset,
    batch_size=10, shuffle=False, num_workers=2
)





# pretrain models
class ShopeeImageEmbeddingNet(nn.Module):
    def __init__(self):
        super(ShopeeImageEmbeddingNet, self).__init__()
              
        model = models.resnet50(True)
        model.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        self.model = model
        
    def forward(self, img):        
        out = self.model(img)
        return out

imgmodel = ShopeeImageEmbeddingNet()
imgmodel = imgmodel.to(device)

imagefeat = []
with torch.no_grad():
    for data in tqdm_notebook(imageloader):
        data = data.to(device)
        feat = imgmodel(data)
        feat = feat.reshape(feat.shape[0], feat.shape[1])
        feat = feat.data.cpu().numpy()
        
        imagefeat.append(feat)

from sklearn.preprocessing import normalize

# l2 norm to kill all the sim in 0-1
imagefeat = np.vstack(imagefeat)
imagefeat = normalize(imagefeat)

print(np.dot(imagefeat[2], imagefeat.T)[:10])
