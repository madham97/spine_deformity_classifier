
import torch
import torch.nn as nn
from torchvision import models

class SpineClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(SpineClassifier, self).__init__()
        # Load a pretrained ResNet model
        self.model = models.resnet50(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
