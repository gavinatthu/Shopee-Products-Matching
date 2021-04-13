import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

def read_sort(DATA_PATH):
    train = pd.read_csv(DATA_PATH + 'train.csv')
    train['image'] = DATA_PATH + 'train_images/' + train['image']
    tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
    train['target'] = train.label_group.map(tmp)
    train = train.sort_values(by='label_group')
    return train


class Dataloader():
    def __init__(self, PATH, height, width):
        self.img_path = PATH
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.img_path)
