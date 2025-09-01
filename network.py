import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # 入力が3chなのはRGB入力だから
        self.bn1 = nn.BatchNorm2d(32) # バッチ正規化
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)  # 画像サイズを半分に落とす
        self.fc1 = nn.Linear(128*16*16, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Conv1 + BN + ReLU
        x = self.pool(x) # 128*128 to 64*64
        
        x = F.relu(self.bn2(self.conv2(x)))  # Conv2 + BN + ReLU
        x = self.pool(x) # 64*64 to 32*32
        
        x = F.relu(self.bn3(self.conv3(x)))  # Conv3 + BN + ReLU
        x = self.pool(x) # 32*32 to 16*16
        
        x = x.view(x.size(0), -1)            # Flatten
        x = F.relu(self.fc1(x))              # FC層
        x = self.fc2(x)
        return x

# Model instantiation (for testing purposes)
if __name__ == "__main__":
    model = CNN()
    print(model)