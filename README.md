
## 模式识别大作业(Shopee-Products-Matching)  
### 功能说明  
data_loader.py 从train.csv中读取图片path，读取图片并转换成tensor，方便torch接口  
p_hash.py 生成图片的感知哈希值，并与原始数据比较  
test.py 一个利用预训练的Resnet50提取图片特征的demo  
### TODO  
面向对象的方法搭建一个Method.py文件 把不同种特征提取方案(KNN, SIFT, CNN等)封装在各自的库中，统一接口  
