import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class SplitResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        self.pretrained = pretrained

        if self.pretrained:
            print("Loading PRE-TRAINED ResNet18 (ImageNet)...")
            weights = ResNet18_Weights.IMAGENET1K_V1
            base_model = resnet18(weights=weights)

            self.G = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
                base_model.avgpool,
            )

            self.F = nn.Identity()
            self.fc = nn.Linear(512, num_classes)

            nn.init.xavier_uniform_(self.fc.weight)

        else:
            print("Initializing SCRATCH ResNet18 (CIFAR Optimized)...")
            base_model = resnet18(weights=None)

            base_model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            base_model.maxpool = nn.Identity()

            self.G = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
            )

            self.F = nn.Sequential(
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
                base_model.avgpool,
                nn.Flatten(),
            )

            self.fc = nn.Linear(512, num_classes)

    def forward_G(self, x):
        if self.pretrained and x.shape[-1] < 64:
            x = torch.nn.functional.interpolate(
                x, size=(64, 64), mode="bilinear", align_corners=False
            )

        return self.G(x)

    def forward_F(self, z):
        if self.pretrained:
            z = z.view(z.size(0), -1)
            return self.fc(z)
        else:
            z = self.F(z)
            return self.fc(z)

    def forward(self, x):
        z = self.forward_G(x)
        return self.forward_F(z)
