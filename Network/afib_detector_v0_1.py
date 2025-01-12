import torch.nn as nn
version = 2     # 1 - nasza implementacja sieci (50% ACC), 2 - implementacja sieci z artykułu(???% ACC)
class AfibDetector(nn.Module):
    def __init__(self):
        super(AfibDetector, self).__init__()
        match version:
            case 1:
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
            case 2:
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
                    nn.Linear(32 * 4 * 4, 100),
                    nn.ReLU()
                )
        
                self.dropout = nn.Dropout(0.25)

                self.fc2 = nn.Sequential(
                nn.Linear(100, 2),
                    nn.Softmax(1)
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
