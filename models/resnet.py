import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, resnet18, resnet34
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


class ResNet_(nn.Module):
    def __init__(self, model=resnet50, in_ch=30, num_classes=1, weights=None, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.model = model(weights=weights)
        if in_ch != 3:
            self.old_conv1 = self.model.conv1
            self.model.conv1 = nn.Conv2d(in_ch, self.old_conv1.out_channels, kernel_size=7, stride=2, padding=3,
                                         bias=False)
        self.old_fc = self.model.fc
        self.model.fc = nn.Linear(self.old_fc.in_features, num_classes)

    def forward(self, inputs):
        x = self.model(inputs)
        if self.num_classes == 1:
            x = torch.sigmoid(x)
        return x


class ResNet18_(ResNet_):
    def __init__(self, in_ch=30, num_classes=1, pretrained=True, *args, **kwargs):
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
        super().__init__(resnet18, in_ch, num_classes, weights, *args, **kwargs)


class ResNet34_(ResNet_):
    def __init__(self, in_ch=30, num_classes=1, pretrained=True, *args, **kwargs):
        if pretrained:
            weights = ResNet34_Weights.DEFAULT
        else:
            weights = None
        super().__init__(resnet34, in_ch, num_classes, weights, *args, **kwargs)


class ResNet50_(ResNet_):
    def __init__(self, in_ch=30, num_classes=1, pretrained=True, *args, **kwargs):
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None
        super().__init__(resnet50, in_ch, num_classes, weights, *args, **kwargs)
