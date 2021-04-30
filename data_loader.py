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


# 改写后的read函数
def read(DATA_PATH):
    origin = pd.read_csv(DATA_PATH + 'train.csv')
    origin = origin.sort_values(by='label_group')
    #print('Items count:', len(origin))
    #print('Classes count:', len(origin['label_group'].value_counts()))  # 总共有11014个类
    train = origin[:30013]   #为了使最后一类完整留在train中向后移位
    test = origin[30013:]
    train_path = DATA_PATH + 'train_images/' + train['image'].values
    test_path = DATA_PATH + 'train_images/' + test['image'].values   #这种切分方法实际上并不优雅，考虑用groupby('label_group')，之后random选取一定比例的组
    # 需要写一个交叉验证分割方法，保证每一类至少有一个点在train中，初步验证先用所有点训练
    # 可以先在每个组中随机抽一个点，剩下的点再随机分割

    return train, test, train_path, test_path


def read_1(DATA_PATH):
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
    return train, test, train_path, test_path

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
