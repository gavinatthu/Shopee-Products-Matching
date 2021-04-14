import pandas as pd
from pandas import Series,DataFrame
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
# 这里参考https://www.kaggle.com/finlay/shopee-products-matching-image-part-english 需要改写

def read(DATA_PATH):
    origin = pd.read_csv(DATA_PATH + 'train.csv')    
    origin = origin.sort_values(by='label_group')
    print(len(origin['label_group'].value_counts()))  # 总共有11014个类
    train = origin[:30013]   #为了使最后一类完整留在train中向后移位
    test = origin[30013:]
    train_path = DATA_PATH + 'train_images/' + train['image'].values
    test_path = DATA_PATH + 'train_images/' + test['image'].values   #这种切分方法实际上并不优雅，考虑用groupby('label_group')，之后random选取一定比例的组
    
    return train, test, train_path, test_path

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

DATA_PATH = '/data/shopee_product_matching/'
BATCH_SIZE = 100
IMG_SIZE = 512

train, test, train_path, test_path = read(DATA_PATH)
imagedataset = Dataloader(train_path[:BATCH_SIZE], IMG_SIZE, IMG_SIZE)
#imagedataset = Dataloader(train['image'].values[:BATCH_SIZE], IMG_SIZE, IMG_SIZE)
imageloader = torch.utils.data.DataLoader(imagedataset, BATCH_SIZE, shuffle=False, num_workers=2)


#print(test.head())
#print(train_path)
