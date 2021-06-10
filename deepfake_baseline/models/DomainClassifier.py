import torch.nn as nn

class DomainClassifier(nn.Module):
    def __init__(self, num_features=2048, num_classes=4):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.relu1(self.fc1(x))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        return out
