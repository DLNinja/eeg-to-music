# Model definitions and loading

import torch
import torch.nn as nn
import torchvision.models as models

DEFAULT_MODEL_PATH = "models/classifiers/EEGResnet_4B/best_model_stft_smooth.pt"

class EEGResnet_4B(nn.Module):
    # Modified 4-block ResNet-18 variant for 4-band EEG emotion classification

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = models.resnet18(weights=None)

        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc      = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EEGResnet_3B(nn.Module):
    # Modified 3-block ResNet-18 variant for 4-band EEG emotion classification

    def __init__(self, num_classes: int = 4):
        super(EEGResnet_3B, self).__init__()

        self.backbone = models.resnet18(weights=None)

        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.backbone.maxpool = nn.Identity()

        # Remove layer4.
        self.backbone.layer4 = nn.Identity()

        # Output of layer3 has 256 channels
        in_features = 256
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class EEGResnet_2B(nn.Module):
    # Modified 2-block ResNet-18 variant for 4-band EEG emotion classification

    def __init__(self, num_classes: int = 4):
        super(EEGResnet_2B, self).__init__()

        self.backbone = models.resnet18(weights=None)

        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.backbone.maxpool = nn.Identity()

        # Remove layer3 and layer4
        self.backbone.layer3 = nn.Identity()
        self.backbone.layer4 = nn.Identity()

        # Output of layer2 has 128 channels.
        in_features = 128
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class EEGResnet_1B(nn.Module):
    # Modified 1-block ResNet-18 variant for 4-band EEG emotion classification

    def __init__(self, num_classes: int = 4):
        super(EEGResnet_1B, self).__init__()

        self.backbone = models.resnet18(weights=None)

        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.backbone.maxpool = nn.Identity()

        # Remove layer2, layer3 and layer4
        self.backbone.layer2 = nn.Identity()
        self.backbone.layer3 = nn.Identity()
        self.backbone.layer4 = nn.Identity()

        # Output of layer1 has 64 channels
        in_features = 64

        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

MODEL_ARCHITECTURES = {
    "EEGResnet_4B": EEGResnet_4B,
    "EEGResnet_3B": EEGResnet_3B,
    "EEGResnet_2B": EEGResnet_2B,
    "EEGResnet_1B": EEGResnet_1B,
}

def load_emotion_model(
    model_path: str = DEFAULT_MODEL_PATH,
    num_classes: int = 4,
    arch_name: str = "EEGResnet_4B",
) -> nn.Module:
    # Load a pre-trained EEGResNet_#B model

    try:
        if arch_name not in MODEL_ARCHITECTURES:
            raise ValueError(f"Unknown architecture '{arch_name}'. Options: {list(MODEL_ARCHITECTURES.keys())}")
        
        model_class = MODEL_ARCHITECTURES[arch_name]
        model      = model_class(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model
    except Exception as e:
        print(f"Warning: Failed to load {arch_name} model from '{model_path}': {e}")
        return None
