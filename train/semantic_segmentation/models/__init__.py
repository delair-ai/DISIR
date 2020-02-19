from .deeplab import DeepLab
from .linknet34 import LinkNet34
from .segnet import SegNet
from .unet import UNet
from .erfnet import ERFNet
from .lednet import LEDNet
from .d3net import D3Net


NETS = {
    "LinkNet34": LinkNet34,
    "SegNet": SegNet,
    "UNet": UNet,
    "DeepLab": DeepLab,
    "ERFNet": ERFNet,
    "LEDNet": LEDNet,
    "D3Net": D3Net,
}
