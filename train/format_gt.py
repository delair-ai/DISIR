"""
Convert RGB label maps into label-encoded ground-truth maps.
"""
from glob import glob
from os.path import join

import click
import cv2
import numpy as np
import rasterio as rio
from PIL import Image
from tqdm import tqdm


@click.command()
@click.option('-d', "--directory", help="Ground-truth directory", type=str, required=True)
@click.option('-n', "--name", help="Dataset name. Only Potsdam or INRIA implemented.", type=str, required=True)
def format_gt(directory, name):
    if name.lower() == "potsdam":
        n_classes = 6
    elif name.lower() == "inria":
        n_classes = 2
    else:
        raise Exception("Only handle Potsdam and INRIA datasets.")
    k = input(
        "This will modify the ground truth maps in your folder. If you don't have a copy of the initial ground\
            truth maps, you should make one. Type\
            'y' to continue, any other key to stop.\n")
    if k != 'y':
        raise Exception("It was another key")
    files = glob(join(directory, '*'))
    for file in tqdm(files, total=len(files)):
        reformat_gt(file, n_classes)
    print("Conversion done.")


def convert_from_color(arr_3d, n_classes):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    if n_classes == 6:
        palette = {0: (255, 255, 255),  # Impervious surfaces (white)
                   1: (0, 0, 255),  # Buildings (blue)
                   2: (0, 255, 255),  # Low vegetation (cyan)
                   3: (0, 255, 0),  # Trees (green)
                   4: (255, 255, 0),  # Cars (yellow)
                   5: (255, 0, 0),  # Clutter (red)
                   6: (0, 0, 0)}  # Undefined (black)

        invert_palette = {v: k for k, v in palette.items()}
        for c, i in invert_palette.items():
            m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i
    elif n_classes == 2:
        arr_2d[np.all(arr_3d == (255, 255, 255), axis=2)] = 1
    elif n_classes == 4:
        palette = {0: (255, 0, 0),  # Impervious surfaces (white)
                   1: (0, 255, 0),  # Buildings (blue)
                   2: (0, 0, 255),  # Low vegetation (cyan)
                   3: (0, 0, 0),  # Undefined (black)
                   }
        invert_palette = {v: k for k, v in palette.items()}
        for c, i in invert_palette.items():
            m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i
    return arr_2d


def reformat_gt(gt, n_classes):
    with rio.open(gt) as src:
        meta = src.meta
        meta.update(dtype=np.uint8, compress="LZW", count=1)
        img = convert_from_color(src.read()[:3].transpose((1, 2, 0)), n_classes)
        height = src.height
        width = src.width
    if meta['crs'] is not None:
        with rio.open(gt, "w", **meta) as out_file:
            out_file.write(img.astype(np.uint8), indexes=1)
    else:
        with rio.open(gt, "w", dtype=np.uint8, compress="LZW", count=1, driver="GTiff", height=height,
                      width=width) as out_file:
            out_file.write(img.astype(np.uint8), indexes=1)
    return

if __name__ == "__main__":
    format_gt()
