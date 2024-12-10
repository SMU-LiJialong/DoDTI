#!/usr/bin/env python
# encoding: utf-8

#****************************
# some functions for training
#***************************

import os
import h5py
import glob
import re
import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt

#%%
def preproc_3d_mask_wt(path_train, path_valid):
    path_bvecs = './data/grdient/bvecs_opt.txt'
    path_bvals = './data/grdient/bvals_opt.txt'
    bvecs = np.loadtxt(path_bvecs, dtype='float32')
    bvals = np.loadtxt(path_bvals, dtype='float32')
    bvals = bvals / 1000.0
    # bval = np.array(1000.0, dtype='float32')

    f = h5py.File(path_train, 'r')
    train_nf = np.array(f['noise free data'], dtype='float32')
    train_n = np.array(f['noise data'], dtype='float32')
    train_s0 = np.array(f['s0 map'], dtype='float32')
    train_d = np.array(f['d map'], dtype='float32')
    train_mask = np.array(f['mask'], dtype='float32')
    f.close()
    # train_s0 = train_s0    # data generator add newaxis
    # train_mask = train_mask

    f = h5py.File(path_valid, 'r')
    valid_nf = np.array(f['noise free data'], dtype='float32')
    valid_n = np.array(f['noise data'], dtype='float32')
    valid_s0 = np.array(f['s0 map'], dtype='float32')
    valid_d = np.array(f['d map'], dtype='float32')
    valid_mask = np.array(f['mask'], dtype='float32')
    f.close()
    # valid_s0 = valid_s0
    # valid_mask = valid_mask

    bvals_train = np.broadcast_to(bvals, (train_nf.shape[0], bvals.shape[0]))
    bvecs_train = np.broadcast_to(bvecs, (train_nf.shape[0], bvecs.shape[0], bvecs.shape[1]))
    bvals_valid = np.broadcast_to(bvals, (valid_nf.shape[0], bvals.shape[0]))
    bvecs_valid = np.broadcast_to(bvecs, (valid_nf.shape[0], bvecs.shape[0], bvecs.shape[1]))

    # -ln()
    # train_s0 = np.clip(train_s0, 1.e-8, 1.0)
    # valid_s0 = np.clip(valid_s0, 1.e-8, 1.0)
    # train_n = np.clip(train_n, 1.e-8, 1.0)
    # valid_n = np.clip(valid_n, 1.e-8, 1.0)
    train_s0 = np.maximum(train_s0, 1e-8)
    valid_s0 = np.maximum(valid_s0, 1e-8)
    train_n = np.maximum(train_n, 1e-8)
    valid_n = np.maximum(valid_n, 1e-8)
    train_lns0 = -1.0 * np.log(train_s0)
    valid_lns0 = -1.0 * np.log(valid_s0)
    train_n = -1.0 * np.log(train_n)
    valid_n = -1.0 * np.log(valid_n)

    train_x = (train_n, bvecs_train, bvals_train, train_mask)
    train_y = np.concatenate([train_lns0, train_d, train_mask], axis=-1)
    valid_x = (valid_n, bvecs_valid, bvals_valid, valid_mask)
    valid_y = np.concatenate([valid_lns0, valid_d, valid_mask], axis=-1)
    return train_x, train_y, valid_x, valid_y


def recon_lls_train(map, bvecs, bval):
    # map  :   one -ln(s0) and six tensor elements    [N,Nx,Ny,Nz,Nq]
    # bvecs:   the diffusion-gradient direction       [Nq,3]
    # bval :   the value of b                         [Nq,1]
    _lns0, dxx, dyy, dzz, dxy, dyz, dxz = tf.split(map, 7, axis=-1)  # [N,Nx,Ny,Nq]
    g1 = bvecs[..., 0]  # [N,1,1,Nq]
    g2 = bvecs[..., 1]
    g3 = bvecs[..., 2]
    fi = dxx * g1 * g1 + dyy * g2 * g2 + dzz * g3 * g3 + 2 * dxy * g1 * g2 + 2 * dyz * g2 * g3 + 2 * dxz * g1 * g3
    signal = _lns0 + bval * fi      # -ln(y)  LLS
    # signal = tf.exp(-1.0 * signal)
    return signal


def recon_dti_test(map, bvecs, bvals):
    # map  :   one s0 and six tensor elements    [N,Nx,Ny,Nq]
    # bvecs:   the diffusion-gradient direction  [Nq,3]
    # bval :   the value of b                    [Nq,1]
    s0, dxx, dyy, dzz, dxy, dyz, dxz = tf.split(map, 7, axis=-1)  # [N,Nx,Ny,Nq]
    bvals = bvals[tf.newaxis, tf.newaxis, tf.newaxis, :]  # [N,1,1,Nq]
    bvecs = bvecs[tf.newaxis, tf.newaxis, tf.newaxis, :, :]  # [N,1,1,Nq,3]
    g1 = bvecs[..., 0]  # [N,1,1,Nq]
    g2 = bvecs[..., 1]
    g3 = bvecs[..., 2]
    fi = dxx * g1 * g1 + dyy * g2 * g2 + dzz * g3 * g3 + 2 * dxy * g1 * g2 + 2 * dyz * g2 * g3 + 2 * dxz * g1 * g3
    s = s0 * tf.exp(-1.0 * bvals * fi)
    return s


def recon_dti_test_ln(map, bvecs, bvals):
    # map  :   one s0 and six tensor elements    [N,Nx,Ny,Nq]
    # bvecs:   the diffusion-gradient direction  [Nq,3]
    # bval :   the value of b                    [Nq,1]
    _lns0, dxx, dyy, dzz, dxy, dyz, dxz = tf.split(map, 7, axis=-1)  # [N,Nx,Ny,Nq]
    bvals = bvals[tf.newaxis, tf.newaxis, tf.newaxis, :]  # [N,1,1,Nq]
    bvecs = bvecs[tf.newaxis, tf.newaxis, tf.newaxis, :, :]  # [N,1,1,Nq,3]
    g1 = bvecs[..., 0]  # [N,1,1,Nq]
    g2 = bvecs[..., 1]
    g3 = bvecs[..., 2]
    fi = dxx * g1 * g1 + dyy * g2 * g2 + dzz * g3 * g3 + 2 * dxy * g1 * g2 + 2 * dyz * g2 * g3 + 2 * dxz * g1 * g3
    _lns = _lns0 + bvals * fi
    return _lns


def range_constraint(x, mask=None):
    _lns0, Db, Ds = tf.split(x, [1, 3, 3], axis=-1)
    _lns0 = tf.clip_by_value(_lns0, -0.5, 20.0)
    # Db = tf.clip_by_value(Db, 0.0, 5.0)
    # Ds = tf.clip_by_value(Ds, -1.0, 1.0)
    Db = tf.clip_by_value(Db, 0.0, 10.0)
    Ds = tf.clip_by_value(Ds, -5.0, 5.0)
    x = tf.concat([_lns0, Db, Ds], axis=-1)
    # x = x * mask
    return x


def initial_para(y, bvecs, bvals, nf, mask):
    # using LLS to fit parameters, and then convert to NLS form
    gx = bvecs[..., 0]  # [N,1,1,Nb]
    gy = bvecs[..., 1]
    gz = bvecs[..., 2]
    dSd_lns0 = tf.ones_like(gx)  # [N,1,1,Nb]
    dSddxx = 1.0 * bvals * gx * gx
    dSddyy = 1.0 * bvals * gy * gy
    dSddzz = 1.0 * bvals * gz * gz
    dSddxy = 2.0 * bvals * gx * gy
    dSddyz = 2.0 * bvals * gy * gz
    dSddxz = 2.0 * bvals * gx * gz
    a = tf.stack([dSd_lns0, dSddxx, dSddyy, dSddzz, dSddxy, dSddyz, dSddxz], axis=-1)  # [N,1,1,Nb,7]
    dims = y.get_shape().as_list()
    dims.append(7)

    # ||AX-Y||
    x_lls = tf.squeeze(tf.linalg.lstsq(tf.broadcast_to(a, dims), y[..., tf.newaxis], l2_regularizer=0.0), axis=-1)
    return x_lls


def findLastCheckPoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.h5'))  # get name list of all .hdf5 files
    # file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).h5.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch
