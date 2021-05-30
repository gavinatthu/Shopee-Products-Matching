
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import gensim
import os
from torch.nn import TransformerEncoder, TransformerEncoderLayer # 实现TransformerEncoder用
from efficientnet_pytorch import EfficientNet
#import timm
# TF_IDF model
class TF_IDF():
    def __init__(self, train_sentences):
        self.train_sentences = train_sentences
        # 建立字典
        self.dictionary = gensim.corpora.Dictionary(self.train_sentences)
        corpus = [self.dictionary.doc2bow(item) for item in self.train_sentences]
        # Tfidf算法 计算tf值
        self.tf = gensim.models.TfidfModel(corpus)

        # 采用LSI算法将TFidf降维到所需维度的特征空间
        self.lsi = gensim.models.LsiModel(corpus, id2word=self.dictionary, num_topics=300)

        # 通过token2id得到特征数（字典里面的键的个数）
        num_features = len(self.dictionary.token2id.keys())
        # 计算稀疏矩阵相似度，建立一个索引
        self.index = gensim.similarities.MatrixSimilarity(self.tf[corpus], num_features=num_features)


    def Feat_ext(self, test_words):  #返回输入test_sentence与全部train_sentences(List)的相似度列表
        new_vec = self.dictionary.doc2bow(test_words)
        vec_tf = self.tf[new_vec]
       
        # 利用lsi算法降维tfidf的特征 
        feat = self.lsi[vec_tf]  #turple list[(0, XXX), (1,XXX)...(299, XXX)]
        feat = [item[1] for item in feat]
        feat = np.array(feat)
        # 某些很短的句子lsi方法会产生<300的向量，暂时的解决方法通过padding实现补齐
        if len(feat)!=300:
            feat = np.pad(feat, (0, 300-len(feat)), 'constant', constant_values = 0)

        # 根据索引矩阵计算相似度 
        #sims = self.index[vec_tf]
        #return feat, sims
        return feat


# Fasttext model
class Fast_Text():
    def __init__(self, train_sentences, MODEL_PATH):
        self.train_sentences = train_sentences
        if os.path.exists(MODEL_PATH):
            print('Loading model...')
            self.model = gensim.models.FastText.load(MODEL_PATH)
        else:
            print('Generating model...')
            self.model = gensim.models.fasttext.load_facebook_model(gensim.test.utils.datapath('/home/common/ljw/shopee/crawl-300d-2M-subword.bin')) #这里需要用绝对路径
            #model = gensim.models.FastText(sentences, vector_size=100, window=20, min_count=0)         #这个是只用本地title训练的FastText
            #model = gensim.models.Word2Vec(sentences, vector_size=100, window=20, min_count=0)         #这个是只用本地title训练的word2vec
            self.model.build_vocab(train_sentences, update=True)
            self.model.train(train_sentences, total_examples=len(train_sentences), epochs=model.epochs)
            self.model.save(MODEL_PATH)
    
    def Feat_ext(self, test_words):
        self.sim_total = []
        feat = None
        for words in test_words:
            if feat is not None:
                feat = feat + self.model.wv[words]
            else:
                feat = self.model.wv[words]
        feat = feat / len(test_words)
        '''
        # 这一步循环非常慢 事实上已经构建了trainsentences的list，可以直接计算sentence和list of sentence的余弦距离
        for i in range(len(self.train_sentences)):
            sim = self.model.wv.n_similarity(test_words,self.train_sentences[i])
            self.sim_total.append(sim)
        return feat, self.sim_total
        '''
        return feat


# word2vec with Glove6B,300d
def word2id(DATA_PATH, BATCH_SIZE, MAX_LENGTH):
    TEXT = torchtext.legacy.data.Field(fix_length=MAX_LENGTH, lower=True)        #划定句子的最大长度为MAX_LENGTH， 小写化！！！！
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
    # 现在考虑不需要迭代器
    train_iter, test_iter = torchtext.legacy.data.BucketIterator.splits(
        (train, test), batch_size=BATCH_SIZE, sort_key=None, shuffle=False, repeat=False)

    return TEXT, LABEL, train, test

def id2vec(pretrained_emb, input):
    input_dim, embedding_dim = pretrained_emb.size()
    embedding = nn.Embedding(input_dim, embedding_dim)
    embedding.weight.data.copy_(pretrained_emb)
    output = embedding(input)
    return output


def sen2list(train, test):
    sentences, train_sentences, train_label, test_sentences, test_label = [], [], [], [], []
    with torch.no_grad():
        for i in range(len(train)):
            sentences.append(train[i].title)
            train_sentences.append(train[i].title)
            train_label.append(train[i].label)
        for i in range(len(test)):
            sentences.append(test[i].title)
            test_sentences.append(test[i].title)
            test_label.append(test[i].label)
    return sentences, train_sentences, train_label, test_sentences, test_label




class W2V(nn.Module):
    def __init__(self, pretrained_emb):
        super(W2V, self).__init__()
        input_dim, embedding_dim = pretrained_emb.size()
        self.word_embeddings = nn.Embedding(input_dim, embedding_dim)

    def forward(self, sentence):
        x = self.word_embeddings(sentence)
        x = torch.mean(x, dim=0)
        return x








# pretrain models
class P_Efnetb5(nn.Module):
    def __init__(self):
        super(ShopeeImageEmbeddingNet, self).__init__()


        #model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        model = EfficientNet.from_pretrained('efficientnet-b5')
        print(model)
        #model = models.resnet50(True)
        #model.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        #model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        model._fc = model._fc = nn.Linear(in_features=2048, out_features=300, bias=True) # 这里将输出特征改为300 方便之后和文本组合
        self.model = model
        self.con1 = nn.Conv1d(1, 1, kernel_size=5,stride=7,padding=25)

    def forward(self, img):        
        out = self.model(img)
        out = torch.squeeze(out)
        out = F.normalize(out, dim=1)
        #out = torch.unsqueeze(out, dim=1)
        #out = self.con1(out)
        #out = torch.squeeze(out)
        return out # Batchsize*2048

# pretrain models
class P_Resnetb5(nn.Module):
    def __init__(self):
        super(ShopeeImageEmbeddingNet, self).__init__()


        model = models.resnet50(True)
        model.eval()
        model._fc = model._fc = nn.Linear(in_features=2048, out_features=300, bias=True) # 这里将输出特征改为300 方便之后和文本组合
        self.model = model
        self.con1 = nn.Conv1d(1, 1, kernel_size=5,stride=7,padding=25)

    def forward(self, img):        
        out = self.model(img)
        out = torch.squeeze(out)
        out = F.normalize(out, dim=1)
        #out = torch.unsqueeze(out, dim=1)
        #out = self.con1(out)
        #out = torch.squeeze(out)
        return out # Batchsize*2048


# 简单CNN处理图片
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 65)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, 33)
        self.fc1 = nn.Linear(16 * 20 * 20, 1920)
        self.fc2 = nn.Linear(1920, 1344)
        self.fc3 = nn.Linear(1344, 11014)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def trainLoop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        model.train()
        pred = model(X)
        loss = loss_fn(pred, torch.max(y, 1)[1]) # CrossEntropyLoss()不支持onehot，只支持表示类别的数

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def testLoop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, torch.max(y, 1)[1]).item()
            correct += (pred.argmax(1) == torch.max(y, 1)[1]).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# 耦合图像和文本的网络
class embeddingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 33)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, 17)
        self.fc1 = nn.Linear(16 * 10 * 10, 800)
        self.fc2 = nn.Linear(800, 600)
        self.tfc = nn.Linear(300, 600)
        self.fc3 = nn.Linear(600, 11014)

    def forward(self, text, img):
        text = F.relu(self.tfc(text))
        img = self.pool(F.relu(self.conv1(img)))
        img = self.pool(F.relu(self.conv2(img)))
        img = torch.flatten(img, 1)
        img = F.relu(self.fc1(img))
        img = F.relu(self.fc2(img))
        x = torch.add(img, text)
        x = self.fc3(x)
        return x

class embeddingNetwork2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 33)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, 17)
        self.fc1 = nn.Linear(16 * 10 * 10, 800)
        self.fc2 = nn.Linear(800, 600)
        self.tfc = nn.Linear(300, 600)
        self.fc3 = nn.Linear(600, 11014)

    def forward(self, text, img):
        text = F.relu(self.tfc(text))
        img = self.pool(F.relu(self.conv1(img)))
        img = self.pool(F.relu(self.conv2(img)))
        img = torch.flatten(img, 1)
        img = F.relu(self.fc1(img))
        img = F.relu(self.fc2(img))
        x = torch.add(img, text)
        x = F.relu(self.fc3(x))
        return x

def trainLoopti(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (text, img, y) in enumerate(dataloader):
        text, img, y = text.to(device), img.to(device), y.to(device)

        # Compute prediction error
        model.train()
        pred = model(text.float(), img)
        loss = loss_fn(pred, torch.max(y, 1)[1]) # CrossEntropyLoss()不支持onehot，只支持表示类别的数

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(img)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def testLoopti(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for text, img, y in dataloader:
            text, img, y = text.to(device), img.to(device), y.to(device)
            pred = model(text.float(), img)
            test_loss += loss_fn(pred, torch.max(y, 1)[1]).item()
            correct += (pred.argmax(1) == torch.max(y, 1)[1]).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
