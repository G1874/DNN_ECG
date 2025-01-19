import torch.nn as nn


class AfibDetector(nn.Module):
    def __init__(self):
        super().__init__() 
    
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2,2)),
            nn.BatchNorm2d(32)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2,2)),
            nn.BatchNorm2d(64)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2,2)),
            nn.BatchNorm2d(128)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(16*16*128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Sequential(
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x