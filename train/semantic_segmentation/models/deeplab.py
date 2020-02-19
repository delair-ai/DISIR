"""
Source: https://github.com/jfzhang95/pytorch-deeplab-xception
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_segmentation.models.backbone import build_backbone
from semantic_segmentation.models.utils.aspp import build_aspp
from semantic_segmentation.models.utils.batchnorm import SynchronizedBatchNorm2d
from semantic_segmentation.models.utils.decoder import build_decoder
from semantic_segmentation.models.utils.jpu import JPU


class DeepLab(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        pretrain=False,
        lf=False,
        backbone="xception",
        output_stride=32,
        sync_bn=True,
        freeze_bn=False,
    ):
        super(DeepLab, self).__init__()
        self.net_name = "DeepLab"
        in_channels += n_classes
        if backbone == "drn":
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(in_channels, backbone, output_stride, BatchNorm)
        if output_stride < 32:
            self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        else:
            if backbone == "mobilenet":
                self.jpu = JPU([32, 96, 320], width=64)
            elif backbone == "xception":
                self.jpu = JPU([256, 728, 2048], width=64)
            else:
                raise NotImplementedError
        self.decoder = build_decoder(n_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, feats = self.backbone(input)
        if isinstance(feats, list):
            low_level_feat, jpu_feats = feats
            jpu_feats.append(x)
        else:
            low_level_feat = feats
        if hasattr(self, "jpu"):
            x = self.jpu(jpu_feats)
        else:
            x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode="bilinear", align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (
                    isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], SynchronizedBatchNorm2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (
                    isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], SynchronizedBatchNorm2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(in_channels=3, n_classes=21, backbone="mobilenet", output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
