import torch
import torch.nn as nn
import torch.nn.functional as F

class CTNet(nn.Module):
    def __init__(self):
        super(CTNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.fc1 = nn.Linear(256*2*2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 5)  # Classes 5-9

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Output: 32x32x32
        x = F.max_pool2d(x, 2)     # Output: 16x16x32
        x = F.relu(self.conv2(x))  # Output: 16x16x64
        x = F.max_pool2d(x, 2)     # Output: 8x8x64
        x = F.relu(self.conv3(x))  # Output: 8x8x128
        x = F.max_pool2d(x, 2)     # Output: 4x4x128
        x = F.relu(self.conv4(x))  # Output: 4x4x256
        x = F.max_pool2d(x, 2)     # Output: 2x2x256
        x = torch.flatten(x, 1)    # Flatten: 2*2*256 = 1024
        x = F.relu(self.fc1(x))    # Output: 512
        x = F.relu(self.fc2(x))    # Output: 256
        x = self.fc3(x)            # Output: 5 (classes 5-9)
        return x
