
## 模式识别大作业(Shopee-Products-Matching)  
### 功能说明  
data_loader.py 从train.csv中读取图片path，读取图片并转换成tensor，方便torch接口  
Methods.py 面向对象地实现各种图像特征提取方法  
p_hash.py 生成图片的感知哈希值，并与原始数据比较  
test.py 一个利用预训练的Resnet50提取图片特征的demo  

### TODO  
搭建一个Methods.py文件 把不同种特征提取方案(KNN, SIFT, CNN等)封装在各自的库中，统一接口  
图片特征提取的pipeline  
  
### 系统环境  
Python 3.8.8  
Pytorch 1.8.1  
CUDA 11.1  
Torchvision 0.9.1  
Torchtext 0.9.1  
gensim 4.0.1  
Pandas 1.2.3  
Numpy 1.19.5  

### 方法  
```bash
$ conda activate ljw_py38  
$ cd ljw/shopee  
$ python demo_text.py
```

### 实验记录  
文本:  
Methods|Top1|Top5|Top20|Time(s)
---|:--:|:--:|:--:|---:
Tf-idf|67.8%|85.3%|92.9%|239.0
Fasttext|45.5%|55.7%|63.2%|16415.9

---
图片:  
Methods|Top1|Top5|Top20|Time(s)
---|:--:|:--:|:--:|---:
ResNet50|68.4%|77.7%|83.3%|353.2
Efficientnet-B5|69.0%|78.6%|84.37%|361.6
SIFT|-|-|-|-
