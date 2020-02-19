from os import path

import click
import numpy as np
import torch

from semantic_segmentation.params import config_factory
from semantic_segmentation.trainer import Trainer


@click.command()
@click.option("-m", "--model", type=str, help="Path to model to convert to graph.")
@click.option("-c", "--config", help="Path to  yaml configuration file.")
@click.option("-o", "--output", help="Path to  the output directory.", default="")
def export_graph(model, config, output):
    """
    Export the graph of a model from Pytorch using Segmenter.save_to_jit.
    """
    output_path = output if output else path.dirname(model)
    if not path.exists(output_path):
        print("Invalid output directory path")
        exit()
    output_path = path.join(output_path, path.basename(model.replace(".pth", ".pt")))
    cfg = config_factory(config)
    net = Trainer(cfg)
    net.load_weights(model)
    net.net.eval()

    torch_out, dummy_tensor = net.save_to_jit(output_path)
    model = torch.jit.load(output_path)
    jit_out = model(dummy_tensor)
    # Compare the results of the two models up to 3 decimals
    np.testing.assert_almost_equal(
        torch_out.data.cpu().numpy(), jit_out.data.cpu().numpy(), decimal=3
    )
    print("Graph succesfuly imported.")


if __name__ == "__main__":
    export_graph()
