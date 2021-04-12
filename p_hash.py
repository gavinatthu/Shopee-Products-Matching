import pandas as pd
import imagehash
import torch
from PIL import Image
import os

os.environ["CUDA_VISIBLE_deviceS"] = "5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = '/data/shopee_product_matching/'

test = pd.read_csv(DATA_PATH + 'test.csv')
print(test.head())

highfreq_factor = 1
hash_size = 32
img_size = hash_size * highfreq_factor

hash1 = imagehash.phash(Image.open(DATA_PATH + 'test_images/0008377d3662e83ef44e1881af38b879.jpg'),hash_size=hash_size,highfreq_factor=highfreq_factor)
print(hash1)
