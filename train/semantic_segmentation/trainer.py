import logging
import os
import time
from typing import List, Tuple

import GPUtil
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from semantic_segmentation.models import NETS


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(
            "cuda:{}".format(self._set_device()) if torch.cuda.is_available() else "cpu"
        )
        print("Using Device: {}".format(self.device))

        params = {}
        self.net = self._get_net(cfg.NET_NAME)(
            in_channels=cfg.IN_CHANNELS, n_classes=cfg.N_CLASSES, **params
        )
        self.net = self.net.to(self.device)
        self.net_name = self.cfg.NET_NAME
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.OPTIM_BASELR,
            momentum=0.9,
            weight_decay=0.0005,
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, list(self.cfg.OPTIM_STEPS), gamma=0.1
        )

    def train(self, epochs, train_ids, test_ids, means, stds, pretrain_file=None):
        pass

    def test(self, test_ids, means, stds, sparsity=0, stride=None):
        pass

    def _get_net(self, net_name: str) -> torch.Tensor:
        return NETS[net_name]

    @staticmethod
    def _set_device():
        """Set gp device when cuda is activated. If code runs with mpi, """
        for d, i in enumerate(GPUtil.getGPUs()):
            if i.memoryUsed < 2500:  # ie:  used gpu memory<900Mo
                device = d
                break
            else:
                if d + 1 == len(GPUtil.getGPUs()):
                    raise Exception("All GPUs are currently used by external program.")
        return device

    def _save_net(
        self, epoch: int, accu, iou, f1, train_loss, test_loss, losses, temp=True
    ) -> None:
        state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "state_dict": self.net.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "epoch": epoch + 1,
            "losses": losses,
            "accu": accu,
            "iou": iou,
            "f1": f1,
        }
        path = (
            os.path.join(self.cfg.PATH_MODELS, "temp") if temp else self.cfg.PATH_MODELS
        )
        if not os.path.isdir(path):
            os.mkdir(path)
        dataset = os.path.basename(self.dataset)

        torch.save(
            state,
            "_".join(
                [
                    os.path.join(path, self.net_name),
                    dataset,
                    f"epoch{epoch}{self.cfg.ext}.pth",
                ]
            ),
        )

    def load_weights(self, path_weights: str) -> None:
        """Only to infer (doesn't load scheduler and optimizer state)."""
        print(path_weights)
        checkpoint = torch.load(path_weights, map_location=str(self.device))
        self.net.load_state_dict(checkpoint["state_dict"])
        logging.info(
            "%s Weights loaded", time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime())
        )

    def save_to_jit(self, name):
        self.net = self.net.to("cpu")
        n_channels = self.cfg.IN_CHANNELS + self.cfg.N_CLASSES
        dummy_tensor = torch.randn(
            (self.cfg.BATCH_SIZE, n_channels, *self.cfg.WINDOW_SIZE)
        )
        self.net.eval()
        torch_out = self.net(dummy_tensor)
        traced_script_module = torch.jit.trace(self.net, dummy_tensor)
        traced_script_module.save(name)
        return torch_out, dummy_tensor
