import pandas as pd
from pandas import Series,DataFrame
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

def buildLabelMAP(DATA_PATH):
    """Build a Look-up-table between labels and their indices

    Map the labels to index (range(0, len(labels)))

    Args: 
        DATA_PATH: data path

    Returns:
        labelMAP: pd Dataframe
        Call labelMAP.iloc[index] to get the label
        Call labelMAP.index[pd.Index(labelMAP).get_loc(label)]  to get the index
    """
    origin = pd.read_csv(DATA_PATH + 'train.csv')
    labelMAP = origin.drop_duplicates(subset=['label_group'],keep='first').iloc[:, 4].reset_index(drop=True)
    return labelMAP

def label2SN(labelMAP, label):
    return labelMAP.index[pd.Index(labelMAP).get_loc(label)]

def SN2label(labelMAP, index):
    return labelMAP.iloc[index]


def read_only(DATA_PATH):
    train = pd.read_csv(DATA_PATH + 'train_set.csv')
    train_path = DATA_PATH + 'train_images/' + train['image'].values
    train_label = train['label_group'].values
    test = pd.read_csv(DATA_PATH + 'test_set.csv')
    test_path = DATA_PATH + 'train_images/' + test['image'].values
    test_label = test['label_group'].values
    return train_path, test_path, train_label, test_label



def read_1(DATA_PATH):
    labelMAP = buildLabelMAP(DATA_PATH)
    origin = pd.read_csv(DATA_PATH + 'train.csv')
    #origin = origin.sort_values(by='label_group')
    print('Len of original set:', len(origin))                               # 总共有34250个样本
    print('Num of classes:', len(origin['label_group'].value_counts()))      # 总共有11014个类
    train_sub = origin.drop_duplicates(subset=['label_group'],keep='first')  # 每个类保留第一个作为训练集的subset
    test_super = origin.append(train_sub)
    test_super = test_super.drop_duplicates(subset=['posting_id'],keep=False)# origin中去除train_subset作为测试集的母集
    test = test_super.sample(n=3000)                                         # 测试集选取3000个样本
    train_super = origin.append(test)
    train = train_super.drop_duplicates(subset=['posting_id'],keep=False)    # 先把test和origin合并，之后去除id重复的项，得到补集
    print('Len of train set:', len(train))
    print('Len of test set:', len(test))
    train_path = DATA_PATH + 'train_images/' + train['image'].values
    test_path = DATA_PATH + 'train_images/' + test['image'].values
    train.to_csv(DATA_PATH + 'train_set.csv', index=False)
    test.to_csv(DATA_PATH + 'test_set.csv', index=False)
    return train, test, train_path, test_path, labelMAP




def read_2(DATA_PATH):
    origin = pd.read_csv(DATA_PATH + 'train.csv')    
    #origin = origin.sort_values(by='label_group')                            # 这行后期去掉
    print('Len of original set:', len(origin))                               # 总共有34250个样本
    print('Num of classes:', len(origin['label_group'].value_counts()))      # 总共有11014个类
    test_super = origin.drop_duplicates(subset=['label_group'],keep='first') # 每个类保留第一个作为测试集的super set
    test = test_super.sample(n=3000)                                         # 测试集选取3000个样本
    train_super = origin.append(test)
    train = train_super.drop_duplicates(subset=['posting_id'],keep=False)    # 先把test和origin合并，之后去除id重复的项，得到补集
    print('Len of train set:', len(train))
    print('Len of test set:', len(test))
    train_path = DATA_PATH + 'train_images/' + train['image'].values
    test_path = DATA_PATH + 'train_images/' + test['image'].values
    train.to_csv(DATA_PATH + 'train_set.csv', index=False)
    test.to_csv(DATA_PATH + 'test_set.csv', index=False)
    return train, test, train_path, test_path



class Dataloader():
    def __init__(self, dataSet, PATH, height, width):
        self.img_path = PATH
        self.dataSet = dataSet
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        img = self.transform(img)
        label = self.dataSet.iloc[index, 4]
        return img, label
    
    def __len__(self):
        return len(self.img_path)

class Dataloader1():
    def __init__(self, dataSet, labelMAP, PATH, height, width):
        self.img_path = PATH
        self.dataSet = dataSet
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.labelMAP = labelMAP
        self.target_transform = transforms.Lambda(lambda y: torch.zeros(11014, dtype=torch.long).scatter_(0, torch.tensor(y), value=1))
        
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        img = self.transform(img)
        label = self.dataSet.iloc[index, 4]
        label = self.target_transform(label2SN(self.labelMAP, label))
        sample = {"image": img, "label": label}
        return img, label # 输出图像和onehot两个tensor

    def __len__(self):
        return len(self.img_path)


class DataloaderImgText():
    def __init__(self, textFeat, dataSet, labelMAP, PATH, height, width):
        self.textFeat = textFeat
        self.img_path = PATH
        self.dataSet = dataSet
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.labelMAP = labelMAP
        self.target_transform = transforms.Lambda(lambda y: torch.zeros(11014, dtype=torch.long).scatter_(0, torch.tensor(y), value=1))
        
    def __getitem__(self, index):
        text = self.textFeat[index]
        img = Image.open(self.img_path[index]).convert('RGB')
        img = self.transform(img)
        label = self.dataSet.iloc[index, 4]
        label = self.target_transform(label2SN(self.labelMAP, label))
        return text, img, label # 输出图像和onehot两个tensor
    
    def __len__(self):
        return len(self.img_path)



'''
# demo of data_loader
DATA_PATH = '../shopee_product_matching/'
BATCH_SIZE = 100
IMG_SIZE = 512

train, test, train_path, test_path = read_1(DATA_PATH)
imagedataset = Dataloader(train_path[:BATCH_SIZE], IMG_SIZE, IMG_SIZE)
#imagedataset = Dataloader(train['image'].values[:BATCH_SIZE], IMG_SIZE, IMG_SIZE)
imageloader = torch.utils.data.DataLoader(imagedataset, BATCH_SIZE, shuffle=False, num_workers=2)
'''
# DATA_PATH = '../shopee_product_matching/'
# labelMAP = buildLabelMAP(DATA_PATH)
# print(labelMAP.index[pd.Index(labelMAP).get_loc(249114794)])
