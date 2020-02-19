from semantic_segmentation.models.backbone import resnet, xception, drn, mobilenet


def build_backbone(in_channels, backbone, output_stride, BatchNorm):
    if backbone == "resnet":
        return resnet.ResNet101(in_channels, output_stride, BatchNorm)
    elif backbone == "xception":
        return xception.AlignedXception(in_channels, output_stride, BatchNorm)
    elif backbone == "drn":
        return drn.drn_d_54(in_channels, BatchNorm)
    elif backbone == "mobilenet":
        return mobilenet.MobileNetV2(in_channels, output_stride, BatchNorm)
    else:
        raise NotImplementedError
