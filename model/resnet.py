import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # Load pre-trained ResNet model trained on ImageNet
        self.model = models.resnet50(pretrained=True)
        
        # Modify the first conv layer to handle CIFAR-100's 32x32 images
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool as it's not needed for small images
        
        # Modify final fully connected layer for CIFAR-100's 100 classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 100)

    def forward(self, x):
        # Get features before the final classification layer
        features = self.model.avgpool(self.model.layer4(
            self.model.layer3(self.model.layer2(
                self.model.layer1(self.model.maxpool(
                    self.model.relu(self.model.bn1(self.model.conv1(x)))))))))
        features = torch.flatten(features, 1)
        
        # Get classification output
        out = self.model.fc(features)
        
        return features, out
