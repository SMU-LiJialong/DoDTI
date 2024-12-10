#!/usr/bin/env python
# encoding: utf-8

#****************************
# Testing simulated data
# To process clinical data with DODTI, please refer to `test_for_clinic_data.py`,
# as certain details require modification.
#***************************

import os
import time
import config
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
import functions_cal_metric as metric
import functions_wlls_3D as func
import matplotlib.pyplot as plt
import functions_tensor_eigen as tso
import dipy.reconst.dti as dti
import net_dodti as admmnet
from dipy.io.image import load_nifti, save_nifti
from dipy.core.gradients import gradient_table
from dipy.denoise.localpca import mppca


# %% ********************************* Setting ***********************************
print('\n\n**Setting parameter')
# config.config_gpu(1)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# PATH
save_img_path = './data/test/simu_result/'
save_metric_path = './data/test/simu_result/'
if not os.path.exists(save_metric_path):
    os.makedirs(save_metric_path)

# Setting
data_name = './exp1_mix' # experiment name
fit_method = 'WLS'       # NLLS, WLS, RT or LLS
slice = 55               # the slice of image for showing
get_pic = True           # save compared pictures

# DoDTI setting
model_epoch = 250        # use the trained model from epoch ?? Here, using the model that we have trained 
num_model = 1            # model number
Ns = 8                   # number of stage for ADMM
Nf = 1                   # number of iteration in fitting block
Nc = 7                   # input image channel(the number of diffusion encoding directions)
recon_flag = False       # whether to reconstruct the DW image from the diffusion tensor field in the output
bn_flag = False          # whether to use batch normalization (BN) layers
mix_precision = False    # whether to use mixed precision
denoiser = 'DnCNN'       # denoiser use DnCNN or ResNet
Nblock = 7               # number of convolutional layers
Nfilters = 64            # number of convolutional kernels
kernel_size = 3          # convolutional kernel size

net_admmnet = admmnet.ADMMNetm(Ns=Ns, Nf=Nf, Na=1, Nblock=Nblock, Nfilters=Nfilters, f=kernel_size,
                               denoiser=denoiser, bn_flag=bn_flag, recon_flag=recon_flag, name='WLLS_'+denoiser)
inpt_b_admmnet = tf.keras.layers.Input(shape=(None, None,  None, Nc), name='input_dti')
inpt_bvec_admmnet = tf.keras.layers.Input(shape=(Nc, 3), name='bvecs')
inpt_bval_admmnet = tf.keras.layers.Input(shape=(Nc,), name='bvals')
inpt_mask_admmnet = tf.keras.layers.Input(shape=(None, None, None, 1), name='mask')
para_map_admmnet = net_admmnet([inpt_b_admmnet, inpt_bvec_admmnet, inpt_bval_admmnet, inpt_mask_admmnet])
model_admmnet = tf.keras.Model(inputs=[inpt_b_admmnet, inpt_bvec_admmnet, inpt_bval_admmnet,
                                       inpt_mask_admmnet], outputs=para_map_admmnet, name='ADMMNet')
model_dir_admmnet = os.path.join(data_name, 'model', 'model_{:04d}.h5'.format(model_epoch))  # for demo
model_admmnet.load_weights(filepath=model_dir_admmnet)
print(model_dir_admmnet)


# %% ********************************* Load data***********************************
#%% load data and gradient information
print('\n** Start loading gradient information and noise free data')
bvecs_path = "./data/gradient/bvecs_opt.txt"
bvals_path = "./data/gradient/bvals_opt.txt"
bvecs = np.loadtxt(bvecs_path)
bvals = np.loadtxt(bvals_path)
gtab = gradient_table(bvals, bvecs)
bvals = bvals/1000.0

df = pd.DataFrame()
for num_data in [1]:  # for demo
    print('Loading data:', num_data)
    num = '%02d' % num_data
    noise_free_path = './data/nf_norm/nf_norm_' + num + '.nii.gz'
    tensor_path = './data/tensor/tensor_cwlls_' + num + '.nii.gz'
    mask_path = './data/mask/mask_' + num + '.nii.gz'
    noise_free, _ = load_nifti(noise_free_path)
    gt_tensor, affine_tensor = load_nifti(tensor_path)
    mask, _ = load_nifti(mask_path)
    noise_free = np.array(noise_free, dtype='float32')
    gt_tensor = np.array(gt_tensor, dtype='float32')
    gt_tensor = gt_tensor * 1000.0

    # ground true tensor eigenvalue decompose
    print('** GT_tensor decomposing...')
    gt_fa, gt_md, gt_cfa, gt_ad, gt_rd = tso.tensor2metric(gt_tensor, mask)
    gt_s0 = noise_free[..., 0]



    # %% ********************************* Load noisy data ***********************************
    for num_sigma in [10, 20, 30, 40]:
        sigma = str(num_sigma)
        # %% Load data
        print('\n\n** Loading noise data...')
        noise_path = './data/rician_norm/rician'+num+'_sigma'+sigma+'.nii.gz'
        noise, affine = load_nifti(noise_path)
        noise = np.array(noise, dtype='float32')
        # Note: Please do not remove the skull from the data before running DoDTI !!!
        print('processing: ', noise_path)
        print('processing: ', noise_free_path)
        # the orientation of input data must be LPS, or flip the image
        print('input data orientation:', nib.orientations.aff2axcodes(affine))



        # %% ********************************* WLLS ***********************************
        # noisy data fitting
        print('** Noisy data are fitting by', fit_method)
        start = time.perf_counter()
        dti_wls = dti.TensorModel(gtab, fit_method=fit_method, return_S0_hat=True)
        tenfit = dti_wls.fit(noise)
        n_s0 = tenfit.S0_hat * mask
        a_evecs = tf.convert_to_tensor(tenfit.evecs)
        a_evals = tf.convert_to_tensor(tenfit.evals)
        a_evals = tf.linalg.diag(a_evals)
        tensor_rc = tf.matmul(a_evecs, tf.matmul(a_evals, tf.transpose(a_evecs, perm=[0, 1, 2, 4, 3]))).numpy()
        tensor_rc = np.reshape(tensor_rc, (tensor_rc.shape[0], tensor_rc.shape[1], tensor_rc.shape[2], 9))
        n_tensor = tensor_rc[:, :, :, (0, 4, 8, 1, 5, 2)] * mask[..., np.newaxis]  # order[Dxx, Dyy, Dzz, Dxy, Dyz, Dxz]
        n_tensor = n_tensor * 1000.0
        n_fa, n_md, n_cfa, n_ad, n_rd = tso.tensor2metric(n_tensor, mask)

        # Time calculating
        end = time.perf_counter()
        time_wlls = round(end-start)
        print("Time taken for WLLS ", time_wlls, 'seconds')



        # %% ********************************* MPPCA + WLLS ***********************************
        # MPPCA denoising
        print('** MPPCA processing')
        start = time.perf_counter()
        denoise1 = mppca(noise, np.array(mask, dtype=bool))
        denoise1 = denoise1 * mask[..., np.newaxis]

        # MPPCA denoised data fitting
        print('** MPPCA denoised data are fitting by', fit_method)
        dti_wls = dti.TensorModel(gtab, fit_method=fit_method, return_S0_hat=True)
        tenfit = dti_wls.fit(denoise1)
        dn1_s0 = tenfit.S0_hat * mask
        a_evecs = tf.convert_to_tensor(tenfit.evecs)
        a_evals = tf.convert_to_tensor(tenfit.evals)
        a_evals = tf.linalg.diag(a_evals)
        tensor_rc = tf.matmul(a_evecs, tf.matmul(a_evals, tf.transpose(a_evecs, perm=[0, 1, 2, 4, 3]))).numpy()
        tensor_rc = np.reshape(tensor_rc, (tensor_rc.shape[0], tensor_rc.shape[1], tensor_rc.shape[2], 9))
        dn1_tensor = tensor_rc[:, :, :, (0, 4, 8, 1, 5, 2)] * mask[..., np.newaxis]
        dn1_tensor = dn1_tensor * 1000.0
        dn1_fa, dn1_md, dn1_cfa, dn1_ad, dn1_rd = tso.tensor2metric(dn1_tensor, mask)

        # Time calculating
        end = time.perf_counter()
        time_mppca = round(end-start)
        print("Time taken for MPPCA ", time_mppca, 'seconds')
        


        # %% ********************************* DoDTI ***********************************
        print('** DoDTI processing')
        # prepare input
        n_dti_input = np.maximum(noise, 1e-8)
        n_dti_input = -1.0 * np.log(n_dti_input)
        n_dti_input = n_dti_input[np.newaxis, ...]
        bvals_test = np.broadcast_to(bvals, (n_dti_input.shape[0], bvals.shape[0]))
        bvecs_test = np.broadcast_to(bvecs, (n_dti_input.shape[0], bvecs.shape[0], bvecs.shape[1]))
        inputs = (n_dti_input, bvecs_test, bvals_test, mask[..., np.newaxis])
        noise = noise * mask[..., np.newaxis]   # for showing the image without background

        # DoDTI Run
        start = time.perf_counter()
        maps_all_stage = model_admmnet(inputs)

        # Cal DTI metrics and restore the scale of tensor
        pred_maps = np.squeeze(maps_all_stage[-1])
        pred_s0 = np.exp(-1.0 * pred_maps[..., 0]) * mask                            # s0
        pred_tensor = pred_maps[..., 1:] * mask[..., np.newaxis]                     # tensor1k
        pred_map = np.concatenate([pred_s0[..., np.newaxis], pred_tensor], axis=-1)  # s0 + tensor1k
        pred_dti = func.recon_dti_test(pred_map, bvecs, bvals)  # need to input s0 and tensor*1000, b_value/1000
        pred_dti = pred_dti * mask[..., np.newaxis]
        pred_fa, pred_md, pred_cfa, pred_ad, pred_rd = tso.tensor2metric(pred_tensor, mask)
        
        # Time calculating
        end = time.perf_counter()
        time_dodti = round(end-start)
        print("Time taken for local DoDTI ", time_dodti, 'seconds')



        # %% ********************************* Calculating ***********************************
        # calculate NRMSE and SSIM, save as excel format
        print('** Calculating and recording NRMSE, SSIM and PSNR')
        # Calculate NRMSE
        nrmse_fa_n = metric.cal_nrmse_mask_ex(gt_fa, n_fa, mask)
        nrmse_fa_dn1 = metric.cal_nrmse_mask_ex(gt_fa, dn1_fa, mask)
        nrmse_fa_pred = metric.cal_nrmse_mask_ex(gt_fa, pred_fa, mask)
        nrmse_md_n = metric.cal_nrmse_mask_ex(gt_md, n_md, mask)
        nrmse_md_dn1 = metric.cal_nrmse_mask_ex(gt_md, dn1_md, mask)
        nrmse_md_pred = metric.cal_nrmse_mask_ex(gt_md, pred_md, mask)
        nrmse_ad_n = metric.cal_nrmse_mask_ex(gt_ad, n_ad, mask)
        nrmse_ad_dn1 = metric.cal_nrmse_mask_ex(gt_ad, dn1_ad, mask)
        nrmse_ad_pred = metric.cal_nrmse_mask_ex(gt_ad, pred_ad, mask)
        nrmse_rd_n = metric.cal_nrmse_mask_ex(gt_rd, n_rd, mask)
        nrmse_rd_dn1 = metric.cal_nrmse_mask_ex(gt_rd, dn1_rd, mask)
        nrmse_rd_pred = metric.cal_nrmse_mask_ex(gt_rd, pred_rd, mask)
        nrmse = np.around([[nrmse_fa_n, nrmse_fa_dn1, nrmse_fa_pred],
                           [nrmse_md_n, nrmse_md_dn1, nrmse_md_pred],
                           [nrmse_ad_n, nrmse_ad_dn1, nrmse_ad_pred],
                           [nrmse_rd_n, nrmse_rd_dn1, nrmse_rd_pred]], decimals=4)
        print('order:\t WLLS, MPPCA-WLLS, DoDTI')
        print('NRMSE_fa:\t ', nrmse[0, :])
        print('NRMSE_md:\t ', nrmse[1, :])
        print('NRMSE_ad:\t ', nrmse[2, :])
        print('NRMSE_rd:\t ', nrmse[3, :])

        # Calculate SSIM
        ssim_fa_n = metric.cal_ssim_mask(gt_fa, n_fa, mask, multichannel=False)
        ssim_fa_dn1 = metric.cal_ssim_mask(gt_fa, dn1_fa, mask, multichannel=False)
        ssim_fa_pred = metric.cal_ssim_mask(gt_fa, pred_fa, mask, multichannel=False)
        ssim_md_n = metric.cal_ssim_mask(gt_md, n_md, mask, multichannel=False)
        ssim_md_dn1 = metric.cal_ssim_mask(gt_md, dn1_md, mask, multichannel=False)
        ssim_md_pred = metric.cal_ssim_mask(gt_md, pred_md, mask, multichannel=False)
        ssim_ad_n = metric.cal_ssim_mask(gt_ad, n_ad, mask, multichannel=False)
        ssim_ad_dn1 = metric.cal_ssim_mask(gt_ad, dn1_ad, mask, multichannel=False)
        ssim_ad_pred = metric.cal_ssim_mask(gt_ad, pred_ad, mask, multichannel=False)
        ssim_rd_n = metric.cal_ssim_mask(gt_rd, n_rd, mask, multichannel=False)
        ssim_rd_dn1 = metric.cal_ssim_mask(gt_rd, dn1_rd, mask, multichannel=False)
        ssim_rd_pred = metric.cal_ssim_mask(gt_rd, pred_rd, mask, multichannel=False)
        ssim = np.around([[ssim_fa_n, ssim_fa_dn1, ssim_fa_pred],
                          [ssim_md_n, ssim_md_dn1, ssim_md_pred],
                          [ssim_ad_n, ssim_ad_dn1, ssim_ad_pred],
                          [ssim_rd_n, ssim_rd_dn1, ssim_rd_pred]], decimals=4)
        print('SSIM_fa:\t ', ssim[0, :])
        print('SSIM_md:\t ', ssim[1, :])
        print('SSIM_ad:\t ', ssim[2, :])
        print('SSIM_rd:\t ', ssim[3, :])

        # Calculate PSNR
        psnr_s0_n = metric.cal_psnr_mask_ex(gt_s0, n_s0, mask)
        psnr_s0_dn1 = metric.cal_psnr_mask_ex(gt_s0, dn1_s0, mask)
        psnr_s0_pred = metric.cal_psnr_mask_ex(gt_s0, pred_s0, mask)
        psnr_dti_n = metric.cal_psnr_mask_ex(noise_free, noise, mask[:, :, :, np.newaxis])
        psnr_dti_dn1 = metric.cal_psnr_mask_ex(noise_free, denoise1, mask[:, :, :, np.newaxis])
        psnr_dti_pred = metric.cal_psnr_mask_ex(noise_free, pred_dti, mask[:, :, :, np.newaxis])
        psnr = np.around([[psnr_s0_n, psnr_s0_dn1, psnr_s0_pred],
                          [psnr_dti_n, psnr_dti_dn1, psnr_dti_pred]], decimals=4)
        print('PSNR_s0: \t ', psnr[0, :])
        print('PSNR_DTI:\t ', psnr[1, :])

        #%% record to excel
        # df = pd.DataFrame()
        name_tag = 'rc' + num + '_sigma' + sigma
        tag_dict = {'Name': [name_tag]}
        res_dict = {
            'Name': ['WLLS', 'MPPCA-WLLS', 'DoDTI'],
            'NRMSE_FA': nrmse[0, :],
            'NRMSE_MD': nrmse[1, :],
            'NRMSE_AD': nrmse[2, :],
            'NRMSE_RD': nrmse[3, :],
            'PSNR_s0': psnr[0, :],
            'PSNE_DTI': psnr[1, :],
            'SSIM_FA': ssim[0, :],
            'SSIM_MD': ssim[1, :],
            'SSIM_AD': ssim[2, :],
            'SSIM_RD': ssim[3, :]}
        df = pd.concat([df, pd.DataFrame(tag_dict)], axis=0)
        df = pd.concat([df, pd.DataFrame(res_dict)], axis=0)
        df.to_excel(save_metric_path + 'different_noise_level_metric.xlsx', index=False)

        # %% ********************************* Plotting ***********************************
        # %%figure show and save
        if get_pic is True:
            # slice = 55
            print('**show and save images')
            fig_s0 = np.vstack([pred_s0[..., slice], dn1_s0[..., slice], n_s0[..., slice], gt_s0[..., slice]])
            fig_s0_res = np.vstack([pred_s0[..., slice] - gt_s0[..., slice],
                                    dn1_s0[..., slice] - gt_s0[..., slice],
                                    n_s0[..., slice] - gt_s0[..., slice],
                                    gt_s0[..., slice] - gt_s0[..., slice]])
            fig_b1k = np.vstack([pred_dti[..., slice, 1], denoise1[..., slice, 1], noise[..., slice, 1], noise_free[..., slice, 1]])
            fig_b1k_res = np.vstack([pred_dti[..., slice, 1] - noise_free[..., slice, 1],
                                    denoise1[..., slice, 1] - noise_free[..., slice, 1],
                                    noise[..., slice, 1] - noise_free[..., slice, 1],
                                    noise_free[..., slice, 1] - noise_free[..., slice, 1]])
            fig_Dxx = np.vstack([pred_tensor[..., slice, 0], dn1_tensor[..., slice, 0], n_tensor[..., slice, 0], gt_tensor[..., slice, 0]])
            fig_Dxx_res = np.vstack([pred_tensor[..., slice, 0] - gt_tensor[..., slice, 0],
                                     dn1_tensor[..., slice, 0] - gt_tensor[..., slice, 0],
                                     n_tensor[..., slice, 0] - gt_tensor[..., slice, 0],
                                     gt_tensor[..., slice, 0] - gt_tensor[..., slice, 0]])
            fig_Dyy = np.vstack([pred_tensor[..., slice, 1], dn1_tensor[..., slice, 1], n_tensor[..., slice, 1], gt_tensor[..., slice, 1]])
            fig_Dyy_res = np.vstack([pred_tensor[..., slice, 1] - gt_tensor[..., slice, 1],
                                     dn1_tensor[..., slice, 1] - gt_tensor[..., slice, 1],
                                     n_tensor[..., slice, 1] - gt_tensor[..., slice, 1],
                                     gt_tensor[..., slice, 1] - gt_tensor[..., slice, 1]])
            fig_Dzz = np.vstack([pred_tensor[..., slice, 2], dn1_tensor[..., slice, 2], n_tensor[..., slice, 2], gt_tensor[..., slice, 2]])
            fig_Dzz_res = np.vstack([pred_tensor[..., slice, 2] - gt_tensor[..., slice, 2],
                                     dn1_tensor[..., slice, 2] - gt_tensor[..., slice, 2],
                                     n_tensor[..., slice, 2] - gt_tensor[..., slice, 2],
                                     gt_tensor[..., slice, 2] - gt_tensor[..., slice, 2]])
            fig_Dxy = np.vstack([pred_tensor[..., slice, 3], dn1_tensor[..., slice, 3], n_tensor[..., slice, 3], gt_tensor[..., slice, 3]])
            fig_Dxy_res = np.vstack([pred_tensor[..., slice, 3] - gt_tensor[..., slice, 3],
                                     dn1_tensor[..., slice, 3] - gt_tensor[..., slice, 3],
                                     n_tensor[..., slice, 3] - gt_tensor[..., slice, 3],
                                     gt_tensor[..., slice, 3] - gt_tensor[..., slice, 3]])
            fig_Dyz = np.vstack([pred_tensor[..., slice, 4], dn1_tensor[..., slice, 4], n_tensor[..., slice, 4], gt_tensor[..., slice, 4]])
            fig_Dyz_res = np.vstack([pred_tensor[..., slice, 4] - gt_tensor[..., slice, 4],
                                     dn1_tensor[..., slice, 4] - gt_tensor[..., slice, 4],
                                     n_tensor[..., slice, 4] - gt_tensor[..., slice, 4],
                                     gt_tensor[..., slice, 4] - gt_tensor[..., slice, 4]])
            fig_Dxz = np.vstack([pred_tensor[..., slice, 5], dn1_tensor[..., slice, 5], n_tensor[..., slice, 5], gt_tensor[..., slice, 5]])
            fig_Dxz_res = np.vstack([pred_tensor[..., slice, 5] - gt_tensor[..., slice, 5],
                                     dn1_tensor[..., slice, 5] - gt_tensor[..., slice, 5],
                                     n_tensor[..., slice, 5] - gt_tensor[..., slice, 5],
                                     gt_tensor[..., slice, 5] - gt_tensor[..., slice, 5]])
            fig_fa = np.vstack([pred_fa[..., slice], dn1_fa[..., slice], n_fa[..., slice], gt_fa[..., slice]])
            fig_fa_res = np.vstack([pred_fa[..., slice] - gt_fa[..., slice],
                                    dn1_fa[..., slice] - gt_fa[..., slice],
                                    n_fa[..., slice] - gt_fa[..., slice],
                                    gt_fa[..., slice] - gt_fa[..., slice]])
            fig_md = np.vstack([pred_md[..., slice], dn1_md[..., slice], n_md[..., slice], gt_md[..., slice]])
            fig_md_res = np.vstack([pred_md[..., slice] - gt_md[..., slice],
                                    dn1_md[..., slice] - gt_md[..., slice],
                                    n_md[..., slice] - gt_md[..., slice],
                                    gt_md[..., slice] - gt_md[..., slice]])
            fig_ad = np.vstack([pred_ad[..., slice], dn1_ad[..., slice], n_ad[..., slice], gt_ad[..., slice]])
            fig_ad_res = np.vstack([pred_ad[..., slice] - gt_ad[..., slice],
                                    dn1_ad[..., slice] - gt_ad[..., slice],
                                    n_ad[..., slice] - gt_ad[..., slice],
                                    gt_ad[..., slice] - gt_ad[..., slice]])
            fig_rd = np.vstack([pred_rd[..., slice], dn1_rd[..., slice], n_rd[..., slice], gt_rd[..., slice]])
            fig_rd_res = np.vstack([pred_rd[..., slice] - gt_rd[..., slice],
                                    dn1_rd[..., slice] - gt_rd[..., slice],
                                    n_rd[..., slice] - gt_rd[..., slice],
                                    gt_rd[..., slice] - gt_rd[..., slice]])
            fig_cfa = np.vstack([pred_cfa[..., slice, :], dn1_cfa[..., slice, :], n_cfa[..., slice, :], gt_cfa[..., slice, :]])
            fig_cfa_res = np.vstack([pred_cfa[..., slice, :] - gt_cfa[..., slice, :],
                                     dn1_cfa[..., slice, :] - gt_cfa[..., slice, :],
                                     n_cfa[..., slice, :] - gt_cfa[..., slice, :],
                                     gt_cfa[..., slice, :] - gt_cfa[..., slice, :]])

            # make background be white, according to vmax
            back_white = np.vstack([mask[..., slice], mask[..., slice], mask[..., slice], mask[..., slice]])
            fig_s0 = np.where(back_white == 0, 0.5, fig_s0)
            fig_b1k = np.where(back_white == 0, 0.3, fig_b1k)
            fig_Dxx = np.where(back_white == 0, 3, fig_Dxx)
            fig_Dyy = np.where(back_white == 0, 3, fig_Dyy)
            fig_Dzz = np.where(back_white == 0, 3, fig_Dzz)
            fig_Dxy = np.where(back_white == 0, 0.5, fig_Dxy)
            fig_Dyz = np.where(back_white == 0, 0.5, fig_Dyz)
            fig_Dxz = np.where(back_white == 0, 0.5, fig_Dxz)
            fig_fa = np.where(back_white == 0, 1, fig_fa)
            fig_md = np.where(back_white == 0, 3, fig_md)
            fig_ad = np.where(back_white == 0, 3, fig_ad)
            fig_rd = np.where(back_white == 0, 3, fig_rd)
            fig_cfa = np.where(np.broadcast_to(back_white[..., np.newaxis], fig_cfa.shape) == 0, 1, fig_cfa)

            # %%
            plt.figure(dpi=600, figsize=(4, 4))
            plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
            plt.subplot(4, 1, 1), plt.imshow(np.rot90(fig_s0, k=3), cmap='gray', vmin=0, vmax=0.5), plt.axis('off')
            plt.subplot(4, 1, 2), plt.imshow(np.rot90(fig_s0_res, k=3), cmap='bwr', vmin=-0.05, vmax=0.05), plt.axis('off')
            plt.subplot(4, 1, 3), plt.imshow(np.rot90(fig_b1k, k=3), cmap='gray', vmin=0, vmax=0.3), plt.axis('off')
            plt.subplot(4, 1, 4), plt.imshow(np.rot90(fig_b1k_res, k=3), cmap='bwr', vmin=-0.05, vmax=0.05), plt.axis('off')
            plt.savefig(os.path.join(save_metric_path, 'rc'+num+'_sigma'+sigma+'_slice' + str(slice) + '_DTI.png'))
            plt.show()
            plt.close()

            plt.figure(dpi=600, figsize=(4, 12))
            plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
            plt.subplot(12, 1, 1), plt.imshow(np.rot90(fig_Dxx, k=3), cmap='gray', vmin=0, vmax=3), plt.axis('off')
            plt.subplot(12, 1, 2), plt.imshow(np.rot90(fig_Dxx_res, k=3), cmap='bwr', vmin=-0.3, vmax=0.3), plt.axis('off')
            plt.subplot(12, 1, 3), plt.imshow(np.rot90(fig_Dyy, k=3), cmap='gray', vmin=0, vmax=3), plt.axis('off')
            plt.subplot(12, 1, 4), plt.imshow(np.rot90(fig_Dyy_res, k=3), cmap='bwr', vmin=-0.3, vmax=0.3), plt.axis('off')
            plt.subplot(12, 1, 5), plt.imshow(np.rot90(fig_Dzz, k=3), cmap='gray', vmin=0, vmax=3), plt.axis('off')
            plt.subplot(12, 1, 6), plt.imshow(np.rot90(fig_Dzz_res, k=3), cmap='bwr', vmin=-0.3, vmax=0.3), plt.axis('off')
            plt.subplot(12, 1, 7), plt.imshow(np.rot90(fig_Dxy, k=3), cmap='gray', vmin=-0.5, vmax=0.5), plt.axis('off')
            plt.subplot(12, 1, 8), plt.imshow(np.rot90(fig_Dxy_res, k=3), cmap='bwr', vmin=-0.3, vmax=0.3), plt.axis('off')
            plt.subplot(12, 1, 9), plt.imshow(np.rot90(fig_Dyz, k=3), cmap='gray', vmin=-0.5, vmax=0.5), plt.axis('off')
            plt.subplot(12, 1, 10), plt.imshow(np.rot90(fig_Dyz_res, k=3), cmap='bwr', vmin=-0.3, vmax=0.3), plt.axis('off')
            plt.subplot(12, 1, 11), plt.imshow(np.rot90(fig_Dxz, k=3), cmap='gray', vmin=-0.5, vmax=0.5), plt.axis('off')
            plt.subplot(12, 1, 12), plt.imshow(np.rot90(fig_Dxz_res, k=3), cmap='bwr', vmin=-0.3, vmax=0.3), plt.axis('off')
            plt.savefig(os.path.join(save_metric_path, 'rc'+num+'_sigma'+sigma+'_slice' + str(slice) + '_Tensor.png'))
            plt.show()
            plt.close()

            plt.figure(dpi=600, figsize=(4, 9))
            plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
            plt.subplot(911), plt.imshow(np.rot90(fig_fa, k=3), cmap='gray', vmin=0, vmax=1), plt.axis('off')
            plt.subplot(912), plt.imshow(np.rot90(fig_fa_res, k=3), cmap='bwr', vmin=-0.3, vmax=0.3), plt.axis('off')
            plt.subplot(913), plt.imshow(np.rot90(fig_md, k=3), cmap='gray', vmin=0, vmax=3), plt.axis('off')
            plt.subplot(914), plt.imshow(np.rot90(fig_md_res, k=3), cmap='bwr', vmin=-0.3, vmax=0.3), plt.axis('off')
            plt.subplot(915), plt.imshow(np.rot90(fig_ad, k=3), cmap='gray', vmin=0, vmax=3), plt.axis('off')
            plt.subplot(916), plt.imshow(np.rot90(fig_ad_res, k=3), cmap='bwr', vmin=-0.3, vmax=0.3), plt.axis('off')
            plt.subplot(917), plt.imshow(np.rot90(fig_rd, k=3), cmap='gray', vmin=0, vmax=3), plt.axis('off')
            plt.subplot(918), plt.imshow(np.rot90(fig_rd_res, k=3), cmap='bwr', vmin=-0.3, vmax=0.3), plt.axis('off')
            plt.subplot(919), plt.imshow(np.rot90(fig_cfa, k=3), cmap='gray', vmin=0, vmax=0.2), plt.axis('off')
            plt.savefig(os.path.join(save_metric_path, 'rc'+num+'_sigma'+sigma+'_slice' + str(slice) + '_Metrics.png'))
            plt.show()
            plt.close()
        print('Next one')
print('** Done! **')
