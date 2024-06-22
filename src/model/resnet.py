import torch
from torch import nn

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights


class Resnet(nn.Module):
    def __init__(self, backbone: str = '18', pretrain: bool = True):
        super().__init__()

        assert backbone in ['18', '34', '50', '101', '152']

        model_dict = {
            '18': (resnet18, ResNet18_Weights),
            '34': (resnet34, ResNet34_Weights),
            '50': (resnet50, ResNet50_Weights),
            '101': (resnet101, ResNet101_Weights),
            '152': (resnet152, ResNet152_Weights)
        }

        assert backbone in model_dict, f"Unsupported backbone {backbone}, choose from {list(model_dict.keys())}"

        model_fn, weights = model_dict[backbone]
        backbone = model_fn(weights=weights.DEFAULT if pretrain else None)

        layers = list(backbone.children())
        self.feature_extractor = nn.Sequential(*layers[:-1])
        self.linear = nn.Linear(in_features=layers[-1].in_features, out_features=6, bias=True if layers[-1].bias is not None else None)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)     # keep batch_size
        x = self.linear(x)
        return x

if __name__ == "__main__":
    x = torch.randn((1, 3, 360, 240))   # Batch_size, Channel, Height, Width

    model = Resnet()

    print(model(x).shape)