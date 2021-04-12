import torch
from torch import nn, einsum
import torch.nn.functional as F

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
