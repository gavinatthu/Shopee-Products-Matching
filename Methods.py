import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchtext

# Convlution Network
class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.mp(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mp(x)
        x = F.relu(x)
        # x: 64*320
        x = x.view(in_size, -1) # flatten the tensor
        # x: 64*10
        x = self.fc(x)
        return x

#
'''
class KNN(nn.Module):

class SIFT(nn.Module):
'''

# word2vec with Glove6B,300d
def word2id(DATA_PATH, BATCH_SIZE, MAX_LENGTH):
    TEXT = torchtext.legacy.data.Field(fix_length=MAX_LENGTH)                      #划定句子的最大长度为MAX_LENGTH
    LABEL = torchtext.legacy.data.Field(sequential=False,dtype=torch.long)
    fields = [(None, None), (None, None), (None, None), ('title', TEXT), ('label', LABEL)]
    
    # torchtext读取csv文件
    train, test = torchtext.legacy.data.TabularDataset.splits(
        path=DATA_PATH, train='train_set.csv', test='test_set.csv', format='csv',
        skip_header=True, fields=fields)
    
    # build the vocabulary, glove6B pretrained model lies in DATA_PATH
    TEXT.build_vocab(train, test, vectors=torchtext.vocab.Vectors(name='glove.6B.300d.txt', cache=DATA_PATH))
    LABEL.build_vocab(train, test, vectors=torchtext.vocab.Vectors(name='glove.6B.300d.txt', cache=DATA_PATH))
    #TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))             #能连接外网的情况下，可以在线获取预训练Glove

    # build the Batch Iterator
    train_iter, test_iter = torchtext.legacy.data.BucketIterator.splits(
        (train, test), batch_size=BATCH_SIZE, sort_key=lambda x:MAX_LENGTH, 
        sort_within_batch=True, shuffle=False, repeat=False)

    #print(TEXT.vocab.vectors)
    #print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())
    #print('Label of 0#', train_iter.dataset.examples[0].label)
    #print('Title of 0#', train_iter.dataset.examples[0].title)
    #print(TEXT.vocab.stoi[train_iter.dataset.examples[0].title[0]])
    #print('Title of 0#', TEXT.vocab.vectors[TEXT.vocab.stoi['bag']])
    return TEXT, LABEL, train_iter, test_iter

def id2vec(pretrained_emb, input):
    input_dim, embedding_dim = pretrained_emb.size()
    embedding = nn.Embedding(input_dim, embedding_dim)
    embedding.weight.data.copy_(pretrained_emb)
    output = embedding(input)
    return output


class RNN(nn.Module):
    def __init__(self, pretrained_emb):
        super().__init__()
        #self.bn = nn.BatchNorm1d()
        input_dim, embedding_dim = pretrained_emb.size()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.embedding.weight = nn.Parameter(pretrained_emb)
        self.embedding.weight.data.copy_(pretrained_emb)

    def forward(self, input):
        #x = self.bn(input)
        x = self.embedding(input)
        return x


class LSTM(nn.Module):
    def __init__(self, pretrained_emb):
        super(LSTM, self).__init__()
        input_dim, embedding_dim = pretrained_emb.size()
        self.word_embeddings = nn.Embedding(input_dim, embedding_dim)


    def forward(self, sentence):
        x = self.word_embeddings(sentence)
        x = torch.mean(x, dim=0)

        return x

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

'''
class SIFT():
    def __init__(self):
        super(SIFT, self).__init__()

        
    def forward(self, img):
'''
