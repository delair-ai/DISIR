import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import f1_score as sk_f1


def accuracy(input, target):
    """Computes the total accuracy"""
    return 100 * float(np.count_nonzero(input == target)) / target.size


def IoU(pred, gt, n_classes, all_iou=False):
    """Computes the IoU by class and returns mean-IoU"""
    # print("IoU")
    iou = []
    for i in range(n_classes):
        if np.sum(gt == i) == 0:
            iou.append(np.NaN)
            continue
        TP = np.sum(np.logical_and(pred == i, gt == i))
        FP = np.sum(np.logical_and(pred == i, gt != i))
        FN = np.sum(np.logical_and(pred != i, gt == i))
        iou.append(TP / (TP + FP + FN))
    # nanmean: if a class is not present in the image, it's a NaN
    result = [np.nanmean(iou), iou] if all_iou else np.nanmean(iou)
    return result


def f1_score(pred, gt, n_classes, all=False):
    f1 = []
    for i in range(n_classes):
        if np.sum(gt == i) == 0:
            f1.append(np.NaN)
            continue
        TP = np.sum(np.logical_and(pred == i, gt == i))
        FP = np.sum(np.logical_and(pred == i, gt != i))
        FN = np.sum(np.logical_and(pred != i, gt == i))
        prec = TP / (TP + FP)
        recall = TP / (TP + FN)
        result = 2 * (prec * recall) / (prec + recall)
        f1.append(result)
    result = [np.nanmean(f1), f1] if all else np.nanmean(f1)
    if all:
        flat_pred = pred.reshape(-1)
        flat_gt = gt.reshape(-1)
        f1_weighted = sk_f1(flat_gt, flat_pred, average="weighted")
        result.append(f1_weighted)
    return result
