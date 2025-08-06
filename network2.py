import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*32*32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Conv1 + BN + ReLU
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))  # Conv2 + BN + ReLU
        x = self.pool(x)
        x = x.view(x.size(0), -1)            # Flatten
        x = F.relu(self.fc1(x))              # FC層
        x = self.fc2(x)
        return x

# Model instantiation (for testing purposes)
if __name__ == "__main__":
    model = SimpleCNN()
    print(model)