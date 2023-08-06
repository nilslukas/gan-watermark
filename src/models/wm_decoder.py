import math

import timm
import torch
from torch import nn
from torch.nn.functional import relu
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights


class WatermarkDecoder(nn.Module):
    def __init__(self, bitlen: int, decoder_arch: str):
        """ A watermark decoder trained with transfer learning. """
        super(WatermarkDecoder, self).__init__()
        self.model_type = decoder_arch
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.Lambda(lambda x: (x + 1) / 2),    # from [-1,1] to [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        if decoder_arch == "resnet18":
            base_model1 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.hidden_size = 512
            self.dense = nn.Linear(self.hidden_size, bitlen)
        elif decoder_arch == "resnet50":
            base_model1 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.hidden_size = 2048
            self.dense = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, bitlen)
            )
        elif decoder_arch == "resnet101":
            base_model1 = models.resnet101(weights=ResNet101_Weights.DEFAULT)
            self.hidden_size = 2048
            self.dense = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, bitlen)
            )
        else:
            raise ValueError(decoder_arch)
        self.base_model1 = torch.nn.Sequential(*list(base_model1.children())[:-1])

    def forward(self, image1):
        f1 = self.base_model1(self.preprocess(image1))
        return self.dense(f1.view(-1, self.hidden_size))

