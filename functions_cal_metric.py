#!/usr/bin/env python
# encoding: utf-8

#****************************
# some functions for calculating NRMSE, SSIM...
#***************************

import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity
from sklearn.metrics import r2_score

# *******************Calculate the NRMSE, MSE, MAE after excluding outliers in the mask*******************
def cal_nrmse_mask_ex(y_true, y_pred, mask, percentage=0.999):
    mask = np.broadcast_to(np.array(mask), np.array(y_pred).shape)  # mask broadcast to [N,h,w,7]
    y_pred_index = np.extract(mask > 0, np.array(y_pred))
    y_true_index = np.extract(mask > 0, np.array(y_true))
    value = np.square(np.subtract(y_pred_index, y_true_index))
    index_sort = np.argsort(value)
    index_split = round(len(index_sort) * percentage)
    index_corr = index_sort[0:index_split]
    e = np.sqrt(np.sum(value[index_corr]))
    gt = np.sqrt(np.sum(np.square(y_true_index[index_corr])))
    nrmse = np.divide(e, gt)
    return nrmse


def cal_r2_score_mask_ex(y_true, y_pred, mask, percentage=0.999):
    mask = np.broadcast_to(np.array(mask), np.array(y_pred).shape)  # mask broadcast to [N,h,w,7]
    y_pred_index = np.extract(mask > 0, np.array(y_pred))
    y_true_index = np.extract(mask > 0, np.array(y_true))
    value = np.square(np.subtract(y_pred_index, y_true_index))
    index_sort = np.argsort(value)
    index_split = round(len(index_sort) * percentage)
    index_corr = index_sort[0:index_split]
    y_true_corr = y_true_index[index_corr]
    y_pred_corr = y_pred_index[index_corr]
    score = r2_score(y_true_corr, y_pred_corr)
    return score


def cal_r2_score_ex(y_true, y_pred, percentage=0.999):
    value = np.square(np.subtract(y_true, y_pred))
    index_sort = np.argsort(value)
    index_split = round(len(index_sort) * percentage)
    index_corr = index_sort[0:index_split]
    y_true_corr = y_true[index_corr]
    y_pred_corr = y_pred[index_corr]
    score = r2_score(y_true_corr, y_pred_corr)
    return score


def cal_mse_mask_ex(y_true, y_pred, mask):
    mask = np.broadcast_to(np.array(mask), np.array(y_pred).shape)  # mask broadcast to [N,h,w,7]
    y_pred_index = np.extract(mask > 0, np.array(y_pred))
    y_true_index = np.extract(mask > 0, np.array(y_true))
    value = np.square(np.subtract(y_pred_index, y_true_index))
    index_sort = np.argsort(value)
    index_split = round(len(index_sort) * 0.999)
    index_corr = index_sort[0:index_split]
    mse = np.mean(value[index_corr])
    return mse

def calculate_v1_mae(v1_gt,v1,mask):
    sum = np.sum(mask)
    idx = int(sum)
    mask_idx = np.where(mask==1)
    select_99 = int(sum*0.99)
    v1_gt = v1_gt[mask_idx]
    v1 = v1[mask_idx]
    theta = np.abs(np.arccos(np.abs(np.sum(v1_gt * v1, axis=-1))))
    theta = np.sort(theta)
    theta = theta[0:select_99]
    v1_mae = np.sum(theta)/select_99
    return v1_mae*(180/np.pi)

def cal_mae_mask_ex(y_true, y_pred, mask):
    mask = np.broadcast_to(np.array(mask[..., np.newaxis]), np.array(y_pred).shape)  # mask broadcast to [N,h,w,7]
    y_pred_index = np.extract(mask > 0, np.array(y_pred))
    y_true_index = np.extract(mask > 0, np.array(y_true))
    value = np.abs(np.subtract(y_pred_index, y_true_index))
    index_sort = np.argsort(value)
    index_split = round(len(index_sort) * 0.99)
    index_corr = index_sort[0:index_split]
    mae = np.mean(value[index_corr])
    return mae


def cal_psnr_mask_ex(y_true, y_pred, mask, percentage=0.999):
    mask = np.broadcast_to(np.array(mask), np.array(y_pred).shape)  # mask broadcast to [N,h,w,7]
    y_pred_index = np.extract(mask > 0, np.array(y_pred))
    y_true_index = np.extract(mask > 0, np.array(y_true))
    value = np.square(np.subtract(y_pred_index, y_true_index))
    index_sort = np.argsort(value)
    index_split = round(len(index_sort) * percentage)
    index_corr = index_sort[0:index_split]
    mse = np.mean(value[index_corr])
    psnr = 10.0*np.log(1.0/mse)/np.log(10.0)
    return psnr


def cal_mse_mask(y_true, y_pred, mask):
    mask = np.broadcast_to(np.array(mask), np.array(y_pred).shape)  # mask broadcast to [N,h,w,7]
    y_pred_index = np.extract(mask > 0, np.array(y_pred))
    y_true_index = np.extract(mask > 0, np.array(y_true))
    value = np.square(np.subtract(y_pred_index, y_true_index))
    index_sort = np.argsort(value)
    index_split = round(len(index_sort) * 1.0)
    index_corr = index_sort[0:index_split]
    mse = np.mean(value[index_corr])
    return mse


def cal_mae_mask(y_true, y_pred, mask):
    mask = np.broadcast_to(np.array(mask), np.array(y_pred).shape)  # mask broadcast to [N,h,w,7]
    y_pred_index = np.extract(mask > 0, np.array(y_pred))
    y_true_index = np.extract(mask > 0, np.array(y_true))
    value = np.abs(np.subtract(y_pred_index, y_true_index))
    index_sort = np.argsort(value)
    index_split = round(len(index_sort) * 1.0)
    index_corr = index_sort[0:index_split]
    mae = np.mean(value[index_corr])
    return mae


def cal_psnr_mask(y_true, y_pred, mask):
    mask = np.broadcast_to(np.array(mask), np.array(y_pred).shape)  # mask broadcast to [N,h,w,7]
    y_pred_index = np.extract(mask > 0, np.array(y_pred))
    y_true_index = np.extract(mask > 0, np.array(y_true))
    value = np.square(np.subtract(y_pred_index, y_true_index))
    index_sort = np.argsort(value)
    index_split = round(len(index_sort) * 1.0)
    index_corr = index_sort[0:index_split]
    mse = np.mean(value[index_corr])
    psnr = 10.0*np.log(1.0/mse)/np.log(10.0)
    return psnr


def cal_ssim_mask(y_true, y_pred, mask, multichannel):
    y_true = np.array(y_true * mask, dtype='float64')
    y_pred = np.array(y_pred * mask, dtype='float64')
    ssim = structural_similarity(y_true, y_pred, multichannel=multichannel)
    return ssim


def MetricNrmseS0Mask(y_true, y_pred):
    true_s0, true_d, mask = tf.split(y_true, [1, 6, 1], axis=-1)  # extract mask [N,h,w,1]
    new_pred = y_pred[..., 0:1]
    new_true = true_s0

    index = tf.where(tf.broadcast_to(mask, new_pred.get_shape().as_list()) > 0)  # mask broadcast to [N,h,w,7]
    y_pred_index = tf.gather_nd(new_pred, index)
    y_true_index = tf.gather_nd(new_true, index)
    e = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_pred_index, y_true_index))))
    gt = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_true_index)))
    loss = tf.math.divide_no_nan(e, gt)
    return loss


def MetricNrmseDbMask(y_true, y_pred):
    true_s0, true_Db, true_Ds, mask = tf.split(y_true, [1, 3, 3, 1], axis=-1)  # extract mask [N,h,w,1]
    new_pred = tf.math.multiply(y_pred[..., 1:4], 0.001)
    new_true = true_Db

    index = tf.where(tf.broadcast_to(mask, new_pred.get_shape().as_list()) > 0)  # mask broadcast to [N,h,w,7]
    y_pred_index = tf.gather_nd(new_pred, index)
    y_true_index = tf.gather_nd(new_true, index)
    e = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_pred_index, y_true_index))))
    gt = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_true_index)))
    loss = tf.math.divide_no_nan(e, gt)
    return loss


def MetricNrmseDsMask(y_true, y_pred):
    true_s0, true_Db, true_Ds, mask = tf.split(y_true, [1, 3, 3, 1], axis=-1)  # extract mask [N,h,w,1]
    new_pred = tf.math.multiply(y_pred[..., 4:7], 0.001)
    new_true = true_Ds

    index = tf.where(tf.broadcast_to(mask, new_pred.get_shape().as_list()) > 0)  # mask broadcast to [N,h,w,7]
    y_pred_index = tf.gather_nd(new_pred, index)
    y_true_index = tf.gather_nd(new_true, index)
    e = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_pred_index, y_true_index))))
    gt = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_true_index)))
    loss = tf.math.divide_no_nan(e, gt)
    return loss


def MetricRmseS0Mask(y_true, y_pred):
    true_s0, true_d, mask = tf.split(y_true, [1, 6, 1], axis=-1)  # extract mask [N,h,w,1]
    new_pred = y_pred[..., 0:1]
    new_true = true_s0

    index = tf.where(tf.broadcast_to(mask, new_pred.get_shape().as_list()) > 0)  # mask broadcast to [N,h,w,7]
    y_pred_index = tf.gather_nd(new_pred, index)
    y_true_index = tf.gather_nd(new_true, index)
    e = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_pred_index, y_true_index))))
    return e


def MetricRmseDbMask(y_true, y_pred):
    true_s0, true_Db, true_Ds, mask = tf.split(y_true, [1, 3, 3, 1], axis=-1)  # extract mask [N,h,w,1]
    new_pred = tf.math.multiply(y_pred[..., 1:4], 0.001)
    new_true = true_Db

    index = tf.where(tf.broadcast_to(mask, new_pred.get_shape().as_list()) > 0)  # mask broadcast to [N,h,w,7]
    y_pred_index = tf.gather_nd(new_pred, index)
    y_true_index = tf.gather_nd(new_true, index)
    e = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_pred_index, y_true_index))))
    return e


def MetricRmseDsMask(y_true, y_pred):
    true_s0, true_Db, true_Ds, mask = tf.split(y_true, [1, 3, 3, 1], axis=-1)  # extract mask [N,h,w,1]
    new_pred = tf.math.multiply(y_pred[..., 4:7], 0.001)
    new_true = true_Ds

    index = tf.where(tf.broadcast_to(mask, new_pred.get_shape().as_list()) > 0)  # mask broadcast to [N,h,w,7]
    y_pred_index = tf.gather_nd(new_pred, index)
    y_true_index = tf.gather_nd(new_true, index)
    e = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_pred_index, y_true_index))))
    return e


def cal_rmse_mask(y_true, y_pred, mask):
    mask = np.broadcast_to(np.array(mask), np.array(y_pred).shape)  # mask broadcast to [N,h,w,7]
    y_pred_index = np.extract(mask > 0, np.array(y_pred))
    y_true_index = np.extract(mask > 0, np.array(y_true))
    value = np.square(np.subtract(y_pred_index, y_true_index))
    index_sort = np.argsort(value)  # 对value进行ascend排序后获取index，异常值的索引排在数组后部
    index_split = round(len(index_sort) * 1.0)
    index_corr = index_sort[0:index_split]
    e = np.sqrt(np.sum(value[index_corr]))
    return e