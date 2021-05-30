
## 模式识别大作业(Shopee-Products-Matching)  
### 功能说明  
data_loader.py 数据读取、测试集训练集划分等数据相关的库  
Methods.py 图片和文本的各种特征提取方法实现  
evaluate.py 特征分析、实验结果分析以及统计学相关库函数  
demo_XXX.py 执行函数  
  
### 系统环境  
首先需要新建conda虚拟环境  
```bash
conda create -n shopee_matching python=3.9  
conda activate shopee_matching   
conda install pytorch torchvision torchtext cudatoolkit=11.1  
```
安装关键依赖项：  
```bash
pip install gensium Pandas tqdm opencv-python  
```
### 代码结构  
本项目采用结构化方法面向对象的方法开发，基于python和pytorch实现主要代码功能，并在服务器端(E5, RTX3090)进行训练，所有的特征提取方法封装在Methods.py中，数据读取、测试集训练集划分等数据相关的库封装在data_loader.py中，特征分析、实验结果分析以及统计学相关库函数封装在evaluate.py，执行函数封装在demo_XXX.py中。数据集放在以相对路径表示的文件夹中：  
```python
DATA_PATH = '../shopee_product_matching/'
```

### 图片特征处理  
利用在demo_img.py中根据Pretrained EfficientNet-v5和Pretrained ResNet50进行模型选择： 
```python
imgmodel = P_Efnetb5().to(device)
or imgmodel = P_Resnetb5().to(device)
```
确定了选择的模型之后，使用的默认使用[0]号GPU进行计算：
```bash
python demo_img.py
```
train from Scratch从头开始训练，因为训练Resnet等大型网络需要大量的资源，而且数据集本身过小，所以我们采用LeNet进行训练：
```bash
python demo_leNet.py
```
SIFT特征提取方法：
```bash
python demo_SIFT.py
```

### 文本特征处理  
运行demo_text.py可以同时输出TF_IDF和Fast_Text两种方法的实验结果：
```bash
python demo_text.py
```
利用pretained_BERT进行测试：
```bash
python demo_BERT.py
```
train from Scratch从头开始训练，因为训练BERT等大型网络需要大量的资源，而且数据集本身过小，所以我们采用LeNet进行训练：
```bash
python demo_EsNet.py
```
