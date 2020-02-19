import logging
import os
import time
from glob import glob

import numpy as np
import pandas as pd
import rasterio as rio
import torch
from PIL import Image
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from semantic_segmentation.loaders.loaders import GTDataset, RGBDataset
from semantic_segmentation.trainer import Trainer
from semantic_segmentation.utils.image import (
    from_coord_to_patch,
    grouper,
    sliding_window,
)
from semantic_segmentation.utils.metrics import IoU, accuracy, f1_score


class ClassicTrainer(Trainer):
    def __init__(self, cfg, train=True, dataset=None):
        super(ClassicTrainer, self).__init__(cfg)
        if train:
            self.train_dataset = RGBDataset(dataset, self.cfg)
            self.gt_dataset = GTDataset(dataset, self.cfg, self.train_dataset.train_ids)
            logging.info(
                f"Train ids (len {len(self.train_dataset.imgs)}): {[os.path.basename(i) for i in self.train_dataset.imgs]}"
            )
            self.dataset = dataset
        test_dataset = RGBDataset(dataset, self.cfg, False)
        logging.info(
            f"Test ids (len {len(test_dataset.imgs)}): {[os.path.basename(i) for i in test_dataset.imgs]}"
        )
        self.metrics = pd.DataFrame(
            data={i: [] for i in [os.path.basename(i) for i in test_dataset.imgs]}
        ).T

    def train(self, epochs, pretrain_file=None):
        logging.info(
            "%s INFO: Begin training",
            time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()),
        )

        iter_ = 0

        start_epoch, accu, iou, f1, train_loss, test_loss, losses = self._load_init(
            pretrain_file
        )
        loss_weights = torch.ones(
            self.cfg.N_CLASSES, dtype=torch.float32, device=self.device
        )
        if self.cfg.WEIGHTED_LOSS or self.cfg.REVOLVER_WEIGHTED:
            weights = self.gt_dataset.compute_frequency()
            if self.cfg.REVOLVER_WEIGHTED:
                self.train_dataset.set_sparsifier_weights(weights)
            if self.cfg.WEIGHTED_LOSS:
                loss_weights = (
                    torch.from_numpy(weights).type(torch.FloatTensor).to(self.device)
                )

        train_loader = self.train_dataset.get_loader(
            self.cfg.BATCH_SIZE, self.cfg.WORKERS
        )
        for e in tqdm(range(start_epoch, epochs + 1), total=epochs + 1 - start_epoch):
            logging.info(
                "\n%s Epoch %s",
                time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()),
                e,
            )
            self.net.train()
            steps_pbar = tqdm(
                train_loader, total=self.cfg.EPOCH_SIZE // self.cfg.BATCH_SIZE
            )
            for data in steps_pbar:
                features, labels = data
                self.optimizer.zero_grad()
                features = features.float().to(self.device)
                labels = labels.float().to(self.device)
                output = self.net(features)
                loss = CrossEntropyLoss(loss_weights)(output, labels.long())
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                iter_ += 1
                steps_pbar.set_postfix({"loss": loss.item()})
            train_loss.append(np.mean(losses[-1 * self.cfg.EPOCH_SIZE :]))
            loss, iou_, acc_, f1_ = self.test()
            test_loss.append(loss)
            accu.append(acc_)
            iou.append(iou_ * 100)
            f1.append(f1_ * 100)
            del (loss, iou_, acc_)
            if e % 5 == 0:
                self._save_net(e, accu, iou, f1, train_loss, test_loss, losses)
            self.scheduler.step()
        # Save final state
        self._save_net(epochs, accu, iou, f1, train_loss, test_loss, losses, False)

    def _infer_image(self, stride, *data):
        """infer one image"""
        with torch.no_grad():
            img = data[0]
            pred = np.zeros(img.shape[1:] + (self.cfg.N_CLASSES,))
        for coords in grouper(
            self.cfg.BATCH_SIZE,
            sliding_window(img, step=stride, window_size=self.cfg.WINDOW_SIZE),
        ):
            data_patches = [from_coord_to_patch(x, coords) for x in data]
            data_patches = np.concatenate([*data_patches], axis=1)
            data_patches = torch.from_numpy(data_patches).float().to(self.device)
            outs = self.net(data_patches).data.cpu().numpy()

            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x : x + w, y : y + h] += out
        return pred

    def test(self, sparsity=None, id_class=None):
        """Test the network on images.
        Args:
            sparsity (int): Number of clicks generated per image. Defalut: None
            id_class (int): class id of the newly sampled click.
                            Only used if sparsity >= 0. 
        """
        logging.info(
            "%s INFO: Begin testing",
            time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()),
        )
        self.net.eval()
        loss, acc, iou, f1 = ([], [], [], [])
        test_dataset = RGBDataset(self.dataset, self.cfg, False, sparsity, id_class)
        test_images = test_dataset.get_loader(1, self.cfg.TEST_WORKERS)
        stride = self.cfg.STRIDE
        modif_pxls = 0
        wrong_pxls = 0
        for idx, data in tqdm(
            zip(test_dataset.test_ids, test_images), total=len(test_dataset.test_ids)
        ):
            data = [i.squeeze(0) for i in data]
            img = data[:-1]
            gt = data[-1].cpu().numpy()
            pred = self._infer_image(stride, *img)
            loss.append(
                CrossEntropyLoss()(
                    torch.from_numpy(np.expand_dims(pred.transpose((2, 0, 1)), axis=0)),
                    torch.from_numpy(np.expand_dims(gt, axis=0)).long(),
                ).item()
            )
            pred = np.argmax(pred, axis=-1)
            metric_iou = IoU(
                pred, gt, self.cfg.N_CLASSES, all_iou=(sparsity is not None)
            )
            metric_f1 = f1_score(
                pred, gt, self.cfg.N_CLASSES, all=(sparsity is not None)
            )
            if sparsity is not None:
                metric_iou, all_iou = metric_iou
                metric_f1, all_f1, weighted_f1 = metric_f1
            metric_acc = accuracy(pred, gt)
            acc.append(metric_acc)
            iou.append(metric_iou)
            f1.append(metric_f1)
            if sparsity is not None:
                file_name = os.path.basename(
                    sorted(glob(os.path.join(self.dataset, "gts", "*")))[idx]
                )
                name = os.path.join(
                    self.cfg.SAVE_FOLDER,
                    "tmp",
                    "preds",
                    self.cfg.ext + self.cfg.NET_NAME + file_name,
                )
                self.metrics.loc[file_name, f"{sparsity}_acc"] = metric_acc
                self.metrics.loc[file_name, f"{sparsity}_IoU"] = metric_iou
                self.metrics.loc[file_name, f"{sparsity}_F1"] = metric_f1
                self.metrics.loc[file_name, f"{sparsity}_F1_weighted"] = weighted_f1
                for c, i in enumerate(all_iou):
                    self.metrics.loc[file_name, f"{sparsity}_IoU_class_{c}"] = i
                for c, i in enumerate(all_f1):
                    self.metrics.loc[file_name, f"{sparsity}_F1_class_{c}"] = i
                if os.path.exists(name):
                    old_pred = rio.open(name).read(1)
                    diff = np.sum(old_pred != pred)
                    modif_pxls += diff
                    logging.info("%s: modified pixels: %s", file_name, diff)
                    logging.info("IoU: %s", metric_iou)
                    logging.info("F1: %s", metric_f1)
                    wrongs = np.sum(np.bitwise_and(old_pred != pred, pred != gt))
                    self.metrics.loc[file_name, f"{sparsity}_wrong_pxls"] = wrongs
                    self.metrics.loc[file_name, f"{sparsity}_good_pxls"] = diff - wrongs
                    wrong_pxls += wrongs

                dataset_name = os.path.basename(self.dataset)
                csv_name = "{}_{}{}.csv".format(
                    os.path.join(self.cfg.SAVE_FOLDER, self.cfg.NET_NAME),
                    dataset_name,
                    self.cfg.ext,
                )
                self.metrics.to_csv(csv_name)
                Image.fromarray(pred.astype(np.uint8)).save(name)
        #  Update logger
        if sparsity is not None:
            logging.info(
                "Total modified pixels: %s", modif_pxls / len(test_dataset.test_ids)
            )
            logging.info(
                "Wrong modified pixels: %s", wrong_pxls / len(test_dataset.test_ids)
            )
        logging.info("Mean IoU : " + str(np.nanmean(iou)))
        logging.info("Mean accu : " + str(np.nanmean(acc)))
        logging.info("Mean F1 : " + str(np.nanmean(f1)))
        return np.mean(loss), np.nanmean(iou), np.mean(acc), np.mean(f1)

    def _load_init(self, pretrain_file):
        if pretrain_file:
            checkpoint = torch.load(pretrain_file)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.net.load_state_dict(checkpoint["state_dict"])
            train_loss = checkpoint["train_loss"]
            test_loss = checkpoint["test_loss"]
            start_epoch = checkpoint["epoch"]
            losses = checkpoint["losses"]
            accu = checkpoint["accu"]
            iou = checkpoint["iou"]
            f1 = checkpoint["f1"]
            logging.info(
                "Loaded checkpoint '{}' (epoch {})".format(
                    pretrain_file, checkpoint["epoch"]
                )
            )
        else:
            start_epoch = 1
            train_loss = []
            test_loss = []
            losses = []
            accu = []
            iou = []
            f1 = []
        return start_epoch, accu, iou, f1, train_loss, test_loss, losses
