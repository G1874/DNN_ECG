import torch.nn as nn

# v0.1 - nasza implementacja sieci (50% ACC)
class AfibDetector(nn.Module):
    def __init__(self):
        super(AfibDetector, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.dropout = nn.Dropout(0.25)

        self.dense1 = nn.Sequential(
            nn.Linear(6*6*64, 64),
            nn.ReLU()
        )

        self.dense2 = nn.Sequential(
            nn.Linear(64, 2),
            # nn.Softmax(dim=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x
