import torch.nn as nn


# v0.2 - implementacja sieci z artykułu
class AfibDetector(nn.Module):
    def __init__(self):
        super().__init__() 
    
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 11)),
            nn.ReLU(),
            nn.MaxPool2d((2, 3))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32,(2, 11)),
            nn.MaxPool2d((2, 3))
        )

        self.fc1 = nn.Sequential(
            nn.Linear(32 * 15 * 2, 100),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.25)

        self.fc2 = nn.Sequential(
            nn.Linear(100, 2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x