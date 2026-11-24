
import torch
from torch import nn
from torchvision import transforms

CLASSES = [
    "battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass",
    "metal", "paper", "plastic", "shoes", "trash", "white-glass",
]
IMG_SIZE = 128

def conv_block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
        nn.MaxPool2d(2),
    )

def build_model(n_classes=12):
    return nn.Sequential(
        conv_block(3, 32), conv_block(32, 64), conv_block(64, 128), conv_block(128, 256),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_classes),
    )

def load_model(weights_path, device):
    model = build_model(len(CLASSES))
    state = torch.load(weights_path, map_location=device, weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
