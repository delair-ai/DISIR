"""
Train a network.
"""
import logging
import os
from os.path import join

import click
import numpy as np
import torch
from numpy import random

from semantic_segmentation.classic_trainer import ClassicTrainer
from semantic_segmentation.params import config_factory


cwd = os.getcwd()
print(cwd)


@click.command()
@click.option("-d", "--dataset", help="Dataset on which to train/test.")
@click.option("-c", "--config", default=None, help="Path to yaml configuration file.")
@click.option(
    "-p",
    "--pretrain_file",
    required=False,
    help="Path to a pretrained network",
    default=None,
)
@click.option(
    "--train/--no-train", help="Activate to skip training (ie only to test)", default=True
)
@click.option(
    "-i",
    "--infer",
    help="Enable for inference with simulated clicks after training",
    default=True,
)
def train(dataset, infer, config, pretrain_file, train):
    """Train a semantic segmentation network on GIS datasets."""
    # Set seeds for reproductibility
    cfg = config_factory(config)
    random.seed(42)
    torch.manual_seed(7)
    torch.backends.cudnn.deterministic = True

    output = cfg.SAVE_FOLDER
    model = cfg.PATH_MODELS
    os.makedirs(output, exist_ok=True)
    os.makedirs(model, exist_ok=True)

    # Set logger
    print(dataset)
    dataset_name = os.path.basename(dataset.strip("/"))
    logging.basicConfig(
        format="%(message)s",
        filename="{}_{}{}.log".format(
            join(cfg.SAVE_FOLDER, cfg.NET_NAME), dataset_name, cfg.ext
        ),
        filemode="w",
        level=logging.INFO,
    )
    logging.info("Config : %s ", cfg)
    logging.info("Dataset, %s", dataset_name)

    net = ClassicTrainer(cfg, dataset=dataset)
    print("ae", train)
    if train:
        print("ee")
        net.train(cfg.EPOCHS, pretrain_file=pretrain_file)

    if infer:
        steps = 121
        id_class = 0
        classes = cfg.N_CLASSES
        range_class = np.arange((steps - 1) / classes, steps, (steps - 1) / classes)
        for i in range(steps):
            if len(range_class) > id_class and (i - 1) == range_class[id_class]:
                id_class += 1
            if not train and pretrain_file:
                net.load_weights(pretrain_file)
            logging.info("\nSparsity: %s clicks, class %s", i, id_class)
            net.test(i, id_class=id_class)


if __name__ == "__main__":
    train()
