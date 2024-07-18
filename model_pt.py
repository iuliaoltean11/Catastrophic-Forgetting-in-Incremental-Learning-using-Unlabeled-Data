import torch.nn as nn
import torch.nn.functional as F #activare, pooling

"""
 - Primul layer convoluțional, care primește imagini cu 3 canale de culoare (RGB) și aplică 32 de filtre convoluționale 3x3, cu padding de 1 pentru a păstra dimensiunea imaginii.
 - Al doilea layer convoluțional, care primește 32 de canale de intrare și aplică 64 de filtre 3x3.
 - Al treilea layer convoluțional, care primește 64 de canale de intrare și aplică 128 de filtre 3x3.
 - Primul layer complet conectat (fully connected), care primește un vector de intrare aplatizat de dimensiune 128 * 4 * 4 și produce un vector de dimensiune 256.
 - Al doilea layer complet conectat, care primește un vector de dimensiune 256 și produce un vector de dimensiune num_classes (în acest caz, 5).
 
"""
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):  # 5 clase
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #activare relu
        x = F.max_pool2d(x, 2) # reduce dimensiunea spatială a imaginii.
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1) #Aplatizează ieșirea de la layer-ul convoluțional pentru a putea fi introdusă în layer-ul complet conectat.
        x = F.relu(self.fc1(x)) #  Aplică primul layer complet conectat și funcția ReLU.
        x = self.fc2(x)
        return x
