#!/usr/bin/env python
# encoding: utf-8

#****************************
# input diffusion tensor and mask to calculate FA, MD, Color-FA, AD, and RD.
#***************************

import numpy as np
def tensor2metric(tensor, mask):
    # input tensor: [Nx, Ny, Nz, 6],  order:[Dxx, Dyy, Dzz, Dxy, Dyz, Dxz]
    # input mask: [Nx, Ny, Nz]
    # data format: numpy, float

    # %%tensor6 to tensor 3*3, vector first 3 dimension
    print('Loading diffusion tensor and mask for calculating metrics')
    Dxx, Dyy, Dzz, Dxy, Dyz, Dxz = np.split(np.array(tensor), 6, axis=-1)  # [Nx, Ny, Nz,6]
    Dx = np.concatenate([Dxx, Dxy, Dxz], axis=-1)  # [Nx, Ny, Nz, 3]
    Dy = np.concatenate([Dxy, Dyy, Dyz], axis=-1)
    Dz = np.concatenate([Dxz, Dyz, Dzz], axis=-1)
    tensor9 = np.stack([Dx, Dy, Dz], axis=-2)  # [Nx, Ny, Nz, 3, 3]
    tensor9 = np.reshape(tensor9, [-1, 3, 3])  # [Nx*Ny*Nz, 3, 3]
    mask = np.array(mask)

    # %%cal and sort eigenvalues and eigenvectors
    eigenValues, eigenVectors = np.linalg.eigh(tensor9)  # [num,3], [num,3,3] from small to big
    v3 = eigenValues[:, 0]
    v2 = eigenValues[:, 1]
    v1 = eigenValues[:, 2]         # main
    e1 = eigenVectors[:, :, 2]     # main
    # %% Determine whether the tensor is positive semidefinite
    if np.min(eigenValues) < 0:
        print('Tensor is negative define, number: %d, total number: %d' % (np.size(np.where(v3 < 0)), np.size(v3)))
    else:
        print('Tensor is positive define')

    # %%cal metrics
    ad = v1
    rd = (v2+v3)/2.0
    md = (v1+v2+v3)/3.0
    fa1 = (np.square(v1-md) + np.square(v2-md) + np.square(v3-md)) * 3.0
    fa2 = (np.square(v1) + np.square(v2) + np.square(v3)) * 2.0
    fa = np.sqrt(np.divide(fa1, fa2, out=np.zeros_like(fa1), where=fa2 != 0))
    # fa = np.clip(fa, 0, 1)
    RGB1 = fa * np.abs(np.squeeze(e1[:, 0]))
    RGB2 = fa * np.abs(np.squeeze(e1[:, 1]))
    RGB3 = fa * np.abs(np.squeeze(e1[:, 2]))
    RGB = np.stack([RGB1, RGB2, RGB3], axis=-1)
    RGB = np.where(RGB > np.sqrt(3)/4, np.sqrt(3)/4, RGB)
    RGB = RGB/(np.sqrt(3)/4)

    # %%vector to image
    fa_ = np.reshape(fa, mask.shape) * mask
    md_ = np.reshape(md, mask.shape) * mask
    ad_ = np.reshape(ad, mask.shape) * mask
    rd_ = np.reshape(rd, mask.shape) * mask
    mask3D = np.stack([mask, mask, mask], axis=-1)
    RGB_ = np.reshape(RGB, mask3D.shape) * mask3D   # 3 channel

    print('Finish: Tensor to FA, MD, Color-FA, AD ,RD')
    return fa_, md_, RGB_, ad_, rd_


def mrtrixtensor2metric(tensor, mask):
    # input MRtrix format tensor: [Nx, Ny, Nz, 6],  order:[Dxx, Dyy, Dzz, Dxy, Dyz, Dxz]
    # input mask: [Nx, Ny, Nz]
    # data format: numpy, float

    # %%tensor6 to tensor 3*3, vector first 3 dimension
    print('Loading diffusion tensor and mask for calculating metrics')
    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = np.split(np.array(tensor), 6, axis=-1)  # [Nx, Ny, Nz,6]
    Dx = np.concatenate([Dxx, Dxy, Dxz], axis=-1)  # [Nx, Ny, Nz, 3]
    Dy = np.concatenate([Dxy, Dyy, Dyz], axis=-1)
    Dz = np.concatenate([Dxz, Dyz, Dzz], axis=-1)
    tensor9 = np.stack([Dx, Dy, Dz], axis=-2)  # [Nx, Ny, Nz, 3, 3]
    tensor9 = np.reshape(tensor9, [-1, 3, 3])  # [Nx*Ny*Nz, 3, 3]
    mask = np.array(mask)

    # %%cal and sort eigenvalues and eigenvectors
    eigenValues, eigenVectors = np.linalg.eigh(tensor9)  # [num,3], [num,3,3] from small to big
    v3 = eigenValues[:, 0]
    v2 = eigenValues[:, 1]
    v1 = eigenValues[:, 2]         # main
    e1 = eigenVectors[:, :, 2]     # main
    # %% Determine whether the tensor is positive semidefinite
    if np.min(eigenValues) < 0:
        print('Tensor is negative define, number: %d, total number: %d' % (np.size(np.where(v3 < 0)), np.size(v3)))
    else:
        print('Tensor is positive define')

    # %%cal metrics
    ad = v1
    rd = (v2+v3)/2.0
    md = (v1+v2+v3)/3.0
    fa1 = (np.square(v1-md) + np.square(v2-md) + np.square(v3-md)) * 3.0
    fa2 = (np.square(v1) + np.square(v2) + np.square(v3)) * 2.0
    fa = np.sqrt(np.divide(fa1, fa2, out=np.zeros_like(fa1), where=fa2 != 0))
    # fa = np.clip(fa, 0, 1)
    RGB1 = fa * np.abs(np.squeeze(e1[:, 0]))
    RGB2 = fa * np.abs(np.squeeze(e1[:, 1]))
    RGB3 = fa * np.abs(np.squeeze(e1[:, 2]))
    RGB = np.stack([RGB1, RGB2, RGB3], axis=-1)
    RGB = np.where(RGB > np.sqrt(3)/4, np.sqrt(3)/4, RGB)
    RGB = RGB/(np.sqrt(3)/4)

    # %%vector to image
    fa_ = np.reshape(fa, mask.shape) * mask
    md_ = np.reshape(md, mask.shape) * mask
    ad_ = np.reshape(ad, mask.shape) * mask
    rd_ = np.reshape(rd, mask.shape) * mask
    mask3D = np.stack([mask, mask, mask], axis=-1)
    RGB_ = np.reshape(RGB, mask3D.shape) * mask3D   # 3 channel

    print('Finish: Tensor to FA, MD, Color-FA, AD ,RD')
    return fa_, md_, RGB_, ad_, rd_

