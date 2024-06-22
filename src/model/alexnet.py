import torch
from torch import nn
from torchvision.models import alexnet, AlexNet_Weights

class AlexNet(nn.Module):
    def __init__(self, num_classes:int = 6, pretrain: bool = True):
        super().__init__()
        self.model = alexnet(weights=AlexNet_Weights.DEFAULT if pretrain else None)

        linear = classifier[-1]
        in_features = linear.in_features
        bias = True if linear.bias is not None else False

        classifier[-1] = nn.Linear(in_features, num_classes, bias)
        self.model.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    x = torch.randn(1, 3, 360, 240)
    model = AlexNet()
    print(model(x).shape)