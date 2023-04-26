import numpy as np


def dice_coef(y_true, y_pred, smooth=1.0):

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        np.sum(y_true_f) + np.sum(y_pred_f) + smooth
    )


def jaccard_coef(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (
        np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth
    )


def iou(y_true, y_pred_thresholded):

    y_true_f = y_true.flatten()
    y_pred_thresholded_f = y_pred_thresholded.flatten()
    intersect = np.logical_and(y_true_f, y_pred_thresholded_f)
    union = np.logical_or(y_true_f, y_pred_thresholded_f)
    iou_score = np.sum(intersect) / np.sum(union)
    return iou_score

