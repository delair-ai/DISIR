"""
Pytorch loader
"""
import logging
import os
import warnings
from glob import glob
from typing import Any, Dict, List

import numpy as np
import numpy.random as random
import rasterio
from albumentations import (
    Compose,
    HorizontalFlip,
    HueSaturationValue,
    RandomBrightnessContrast,
    RGBShift,
    ShiftScaleRotate,
    VerticalFlip,
)
from icecream import ic
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from semantic_segmentation.loaders.SegDataLoader import SegDataLoader
from semantic_segmentation.loaders.sparsifier import Sparsifier

rasterio_loader = logging.getLogger("rasterio")
rasterio_loader.setLevel(
    logging.ERROR
)  # rasterio outputs warnings with some tiff files.


class RGBDataset(SegDataLoader):
    """Torch dataloader. Assume that ground-truth files and images have similar filenames.
    Assume the following data structure: DatasetDirectory/imgs and DatasetDirectory/gts where gts 
    contains the ground-truths and imgs the images.
    Read img and gt files. Also read annot and previous pred files to simulate sparsity 
    if needed.
    """

    def __init__(
        self,
        dataset: str,
        cfg: Dict[str, Any],
        train: bool = True,
        test_sparsity: int = None,
        id_class: int = None,
    ):
        """Constructor
        ---------
        Parameters:
            dataset (str): path to the dataset
            cfg (Dict): Config
            train (bool): Different behaviors in train and test. Defaults to True.
            test_sparsity (int): Number of simulated annotations during test. Defaults to None.
            id_class (int): id of the simulated class if test_sparsity>0. Defaults to None.
        """
        super().__init__(dataset, cfg)
        self.test_sparsity = test_sparsity

        self.cfg = cfg
        self.train = train
        self.id_class = id_class
        self.sparsifier = Sparsifier(train, cfg, None)
        self.train_ids, self.test_ids = self.split_dataset(cfg.test_size)

        # gather all files (images and ground-truths + annots & preds if test )
        gts = sorted(glob(os.path.join(dataset, "gts/*")))
        ext_gt = "." + gts[0].split(".")[1]
        ext_imgs = "." + sorted(glob(os.path.join(dataset, "imgs/*")))[0].split(".")[1]
        imgs = [file.replace("gts", "imgs").replace(ext_gt, ext_imgs) for file in gts]
        annot_files = [
            os.path.join(
                self.cfg.SAVE_FOLDER,
                "tmp",
                "annots",
                cfg.ext + cfg.NET_NAME + os.path.basename(file),
            )
            for file in gts
        ]
        pred_files = [
            os.path.join(
                self.cfg.SAVE_FOLDER,
                "tmp",
                "preds",
                cfg.ext + cfg.NET_NAME + os.path.basename(file),
            )
            for file in gts
        ]
        if train:
            self.gts = [gts[i] for i in self.train_ids]
            self.imgs = [imgs[i] for i in self.train_ids]
        else:
            self.gts = [gts[i] for i in self.test_ids]
            self.imgs = [imgs[i] for i in self.test_ids]
        if test_sparsity is not None:
            assert not train
            self.pred_files = [pred_files[i] for i in self.test_ids]
            self.annot_files = [annot_files[i] for i in self.test_ids]
            os.makedirs(os.path.dirname(self.pred_files[0]), exist_ok=True)
            os.makedirs(os.path.dirname(self.annot_files[0]), exist_ok=True)
            if not test_sparsity:
                # ie if it's the first pass, delete old files
                for (i, j) in zip(self.pred_files, self.annot_files):
                    if os.path.exists(i):
                        os.remove(i)
                    if os.path.exists(j):
                        os.remove(j)
        else:
            self.pred_files, self.annot_files = None, None

    def _load_data(self, i):
        """
        if train: Pick a random image and randomly set the coordinates of the crop
        if test: load full images without any randomness
        Load previous annot and pred files during test with simulated annotations
        """
        if self.train:
            random_id = random.randint(len(self.gts))
            with rasterio.open(self.gts[random_id]) as src:
                x_crop, y_crop = (
                    random.randint(max(1, src.shape[1] - self.cfg.WINDOW_SIZE[1])),
                    random.randint(max(1, src.shape[0] - self.cfg.WINDOW_SIZE[0])),
                )
                window = Window(
                    x_crop, y_crop, self.cfg.WINDOW_SIZE[1], self.cfg.WINDOW_SIZE[0]
                )
                del (src, x_crop, y_crop)
        else:
            random_id = i
            window = None

        # Read data
        with rasterio.open(self.imgs[random_id]) as src:
            img = np.asarray(1 / 255 * src.read(window=window), dtype=np.float32)[
                :3
            ].transpose((1, 2, 0))
            features = img
        with rasterio.open(self.gts[random_id]) as src:
            labels = src.read(1, window=window)
        do_load = self.test_sparsity is not None and self.test_sparsity > 0
        previous_pred = (
            rasterio.open(self.pred_files[random_id]).read(1, window=window)
            if do_load
            else None
        )
        previous_annot = (
            rasterio.open(self.annot_files[random_id]).read(1, window=window)
            if do_load
            else None
        )
        return features, labels, previous_pred, previous_annot

    def _data_augmentation(self, features, labels):
        transform = Compose([HorizontalFlip(), VerticalFlip()], p=1)
        transform = transform(image=features, mask=labels)
        features = transform["image"]
        labels = transform["mask"]
        return features, labels

    def __getitem__(self, i):
        """
        Sparsity and augmentation are applied if it was enabled in cfg.
        Returns
        -------
        Data and ground truth in the right tensor shape.
        """
        with warnings.catch_warnings():
            # Remove warnings when image is not georeferenced.
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            features, labels, previous_pred, previous_annot = self._load_data(i)

        #  Data augmentation
        if self.train and self.cfg.TRANSFORMATION:
            features, labels = self._data_augmentation(features, labels)

        # Interactivity
        if self.test_sparsity is not None:
            name = self.annot_files[i]
            name = os.path.join(os.path.dirname(name), os.path.basename(self.gts[i]))
        else:
            name = None
        train_sparsity = random.randint(1, 101) if random.random() > 0.3 else 0
        sparsity = train_sparsity if self.train else self.test_sparsity

        annots = self.sparsifier.simulate_interactivity(
            labels, previous_pred, previous_annot, sparsity, name, self.id_class
        )
        features = np.concatenate([features, annots], axis=2)

        features = features.transpose((2, 0, 1))
        return features, labels

    def set_sparsifier_weights(self, weights):
        self.sparsifier.weights = weights


class GTDataset(SegDataLoader):
    """Only load ground truth. Used to compute classes frequency."""

    def __init__(self, dataset: str, cfg: Dict[str, Any], ids: List[int]):
        self.cfg = cfg
        self.dataset = dataset
        gts = sorted(glob(os.path.join(dataset, "gts/*")))
        self.gts = []
        self.gts = [gts[i] for i in ids]

    def _load_data(self, i):
        with rasterio.open(self.gts[i]) as src:
            labels = src.read(1)
        return labels

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, i):
        # load data
        with warnings.catch_warnings():
            # Remove warnings when image is not georeferenced.
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            labels = self._load_data(i)
        return labels

    def compute_frequency(self):
        print("Computing weights...")
        weights = [[] for i in range(self.cfg.N_CLASSES)]
        labels = self.get_loader(1, 12)
        for gt in labels:
            for i in range(self.cfg.N_CLASSES):
                weights[i].append(np.where(gt == i)[0].shape[0])
        sum_pxls = np.sum(weights)
        weights = [1 / (np.sum(i) / sum_pxls) for i in weights]
        if self.cfg.N_CLASSES == 6:  # assume it's Potsdam dataset
            weights[-1] = min(weights)  # because clutter class is an ill-posed problem
        weights = np.asarray(weights)
        logging.info(f"Following weights have been computed: {weights}")
        ic(weights)
        return weights

    def _data_augmentation(self):
        pass
