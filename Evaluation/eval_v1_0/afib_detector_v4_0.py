import torch.nn as nn


class AfibDetector(nn.Module):
    def __init__(self):
        super().__init__() 
    
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d((2,2)),
            nn.BatchNorm2d(8)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((2,2)),
            nn.BatchNorm2d(16)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2,2)),
            nn.BatchNorm2d(32)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*8*32, 256),
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