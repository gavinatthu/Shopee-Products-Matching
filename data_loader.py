import pandas as pd
from pandas import Series,DataFrame
import torch
import torchvision
import torchvision.transforms as transforms
import torchtext
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


def textLoader(DATA_PATH, BATCH_SIZE):
    TEXT = torchtext.legacy.data.Field()
    LABEL = torchtext.legacy.data.Field(sequential=False,dtype=torch.long)
    fields = [(None, None), (None, None), (None, None), ('title', TEXT), ('label', LABEL)]

    # 这里用torchtext读取csv文件，同样的功能也可以用pandas实现
    train = torchtext.legacy.data.TabularDataset(
        path=DATA_PATH + 'train.csv', format='csv',
        skip_header=True, fields=fields)

    # build the vocabulary, glove6B pretrained model lies in DATA_PATH
    TEXT.build_vocab(train, vectors=torchtext.vocab.Vectors(name='glove.6B.300d.txt', cache=DATA_PATH))

    # build the Batch Iterator
    train_iter = torchtext.legacy.data.BucketIterator(train, batch_size=BATCH_SIZE, sort_key=lambda x:len(x.title), sort_within_batch=True,
                                shuffle=True, repeat=False)

    print('Label of 0#', train_iter.dataset.examples[0].label)
    print('Title of 0#', train_iter.dataset.examples[0].title)
    print(TEXT.vocab.stoi[train_iter.dataset.examples[0].title[0]])
    #print('Title of 0#', TEXT.vocab.vectors[TEXT.vocab.stoi['you']])
    return TEXT, train_iter





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
# 改为相对路径，数据在上层文件夹的shopee-product-matching中
DATA_PATH = '/data/shopee_product_matching/'
BATCH_SIZE = 100
IMG_SIZE = 512

train, test, train_path, test_path = read(DATA_PATH)
imagedataset = Dataloader(train_path[:BATCH_SIZE], IMG_SIZE, IMG_SIZE)
#imagedataset = Dataloader(train['image'].values[:BATCH_SIZE], IMG_SIZE, IMG_SIZE)
imageloader = torch.utils.data.DataLoader(imagedataset, BATCH_SIZE, shuffle=False, num_workers=2)


#print(test.head())
print(train_path)
'''
