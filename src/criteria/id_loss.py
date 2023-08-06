import os.path
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from src.models.model_irse import Backbone
from src.utils.highlited_print import print_warning


class IDLoss(nn.Module):
    def __init__(self, ir_se50_weights):
        super(IDLoss, self).__init__()
        self.ir_se50_weights = ir_se50_weights
        if ir_se50_weights is None or not os.path.exists(ir_se50_weights):
            print_warning(f"No model weights provided for IDLoss. Skipping.")
            return
        print('> Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(ir_se50_weights))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x_hat, x) -> Tuple[float, float]:
        if self.ir_se50_weights is None or not os.path.exists(self.ir_se50_weights):
            return torch.Tensor([0]).to(x_hat.device), torch.Tensor([0]).to(x_hat.device)

        x_hat = F.interpolate(x_hat, size=(256, 256))
        x = F.interpolate(x, size=(256, 256))

        n_samples = x.shape[0]
        y_feats = self.extract_feats(x)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(x_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count, sim_improvement / count
