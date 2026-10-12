"""
MobileFaceNet & ArcFace PyTorch Implementation.
A lightweight deep neural network designed for real-time mobile and edge face recognition.
Extracts 128-D or 512-D L2-normalized face embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """Depthwise Separable Residual Bottleneck block with PReLU."""
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(Bottleneck, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expansion_factor

        layers = []
        if expansion_factor != 1:
            # Pointwise expansion
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.PReLU(hidden_dim))

        # Depthwise convolution
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.PReLU(hidden_dim))

        # Pointwise linear projection (no non-linearity)
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileFaceNet(nn.Module):
    """
    MobileFaceNet Backbone.
    Input shape: (B, 3, 112, 112)
    Output shape: (B, embedding_size) L2-normalized feature vector.
    """
    def __init__(self, embedding_size=128):
        super(MobileFaceNet, self).__init__()
        self.embedding_size = embedding_size

        # Initial stem convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 112x112 -> 56x56
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        self.conv2_dw = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        # Bottlenecks sequence
        self.bottlenecks = nn.Sequential(
            Bottleneck(64, 64, stride=2, expansion_factor=2),     # 56x56 -> 28x28
            Bottleneck(64, 64, stride=1, expansion_factor=2),
            Bottleneck(64, 64, stride=1, expansion_factor=2),
            Bottleneck(64, 64, stride=1, expansion_factor=2),
            Bottleneck(64, 64, stride=1, expansion_factor=2),

            Bottleneck(64, 128, stride=2, expansion_factor=4),    # 28x28 -> 14x14
            Bottleneck(128, 128, stride=1, expansion_factor=2),
            Bottleneck(128, 128, stride=1, expansion_factor=2),
            Bottleneck(128, 128, stride=1, expansion_factor=2),
            Bottleneck(128, 128, stride=1, expansion_factor=2),
            Bottleneck(128, 128, stride=1, expansion_factor=2),

            Bottleneck(128, 128, stride=2, expansion_factor=4),   # 14x14 -> 7x7
            Bottleneck(128, 128, stride=1, expansion_factor=2),
            Bottleneck(128, 128, stride=1, expansion_factor=2),
        )

        # 1x1 conv to 512 channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.PReLU(512)
        )

        # Global Depthwise Convolution (GDConv) 7x7 -> 1x1
        self.gdconv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=7, stride=1, padding=0, groups=512, bias=False),
            nn.BatchNorm2d(512)
        )

        # Linear projection to embedding size
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, normalize=True):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.bottlenecks(x)
        x = self.conv3(x)
        x = self.gdconv(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.bn(x)
        if normalize:
            x = F.normalize(x, p=2, dim=1)
        return x


class ArcMarginProduct(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition (Deng et al., CVPR 2019).
    Projects features and class weights to hypersphere and applies angular margin:
    cos(theta + m) for the ground-truth target class.
    """
    def __init__(self, in_features, out_features, scale=64.0, margin=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = scale
        self.m = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # input: (B, in_features) normalized embeddings
        # weight: (out_features, in_features)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class TripletLoss(nn.Module):
    """
    Triplet Margin Loss for Direct Face Metric Learning.
    L(a, p, n) = max(0, d(a, p) - d(a, n) + margin)
    """
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
