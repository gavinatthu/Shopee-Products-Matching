import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import gensim
import os

# TF_IDF model
class TF_IDF():
    def __init__(self, train_sentences):
        self.train_sentences = train_sentences
        # 建立字典
        self.dictionary = gensim.corpora.Dictionary(self.train_sentences)
        corpus = [self.dictionary.doc2bow(item) for item in self.train_sentences]
        # Tfidf算法 计算tf值
        self.tf = gensim.models.TfidfModel(corpus)
        # 通过token2id得到特征数（字典里面的键的个数）
        num_features = len(self.dictionary.token2id.keys())
        # 计算稀疏矩阵相似度，建立一个索引
        self.index = gensim.similarities.MatrixSimilarity(self.tf[corpus], num_features=num_features)


    def Sim_list(self, test_words):  #返回输入test_sentence与全部train_sentences(List)的相似度列表
        new_vec = self.dictionary.doc2bow(test_words)
        # 根据索引矩阵计算相似度
        sims = self.index[self.tf[new_vec]]
        return sims


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
    
    def Sim_list(self, test_words):
        self.sim_total = []
        # 这一步循环非常慢 事实上已经构建了trainsentences的list，可以直接计算sentence和list of sentence的余弦距离
        for i in range(len(self.train_sentences)):
            sim = self.model.wv.n_similarity(test_words,self.train_sentences[i])
            self.sim_total.append(sim)
        return self.sim_total


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
