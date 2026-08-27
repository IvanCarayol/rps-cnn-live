import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 128x128 -> 64x64, 16 canales
        x = self.pool(self.relu(self.conv2(x)))   # 64x64 -> 32x32, 32 canales

        x = x.view(x.size(0), -1)  # aplanar, manteniendo la dimensión de batch

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x