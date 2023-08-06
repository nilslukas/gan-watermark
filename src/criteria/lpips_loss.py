import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


class LPIPSLoss(nn.Module):
    def __init__(self):
        super(LPIPSLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features.eval().cuda()
        self.layers = [2, 7, 14, 21, 28]
        self.weights = [0.1, 0.1, 0.1, 0.1, 0.6]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        x = self.preprocess(x)
        y = self.preprocess(y)

        x_features = self.get_features(x)
        y_features = self.get_features(y)

        lpips_loss = 0
        for x_feat, y_feat, weight in zip(x_features, y_features, self.weights):
            lpips_loss += weight * (x_feat - y_feat).pow(2).mean().sqrt()

        return lpips_loss

    def preprocess(self, x):
        # Rescale from [-1, 1] to [0, 1]
        x = (x + 1) / 2.0
        x = F.interpolate(x, size=(256, 256))

        # Normalize using ImageNet mean and std
        x = (x - torch.tensor(self.normalize.mean).view(1, 3, 1, 1).cuda()) / torch.tensor(self.normalize.std).view(1,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    1).cuda()

        return x

    def get_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features

