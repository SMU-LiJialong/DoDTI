#!/usr/bin/env python
# encoding: utf-8

#****************************
# Testing clinical data
# Clinical data is not included in this demo.
# However, you can replace the data and gradient table to execute this code.
# Key details are highlighted with *******important*********.
#***************************

import os
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


# %% Setting parameter
print('\n\n**Setting parameter')
# config.config_gpu(1)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

data_name = os.path.join(os.path.abspath(os.getcwd()), 'exp8_mix')
save_img_path = '/public2/lijialong/Data/real/xu_you_sheng/result1/'
if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)


# Setting
data_name = './exp1_mix' # experiment name
fit_method = 'WLS'       # NLLS, WLS, RT or LLS
slice = 50               # the slice of image for showing
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


#%% Load data
print('\n\n** Loading data...')
num_id = 2
dir_num = '30'

path_data = "/public2/lijialong/Data/real/xu_you_sheng/result/raw_"+str(num_id)+"_sel" + dir_num + ".nii.gz"
path_mask = "/public2/lijialong/Data/real/xu_you_sheng/mask_"+str(num_id)+".nii.gz"
mask = nib.load(path_mask).get_fdata().astype('float32')
data, affine = load_nifti(path_data)
data_mask = data * mask[..., np.newaxis]
data = np.clip(data, a_min=0, a_max=np.max(data))
print('processing: ', path_data)
print('orientation:', nib.orientations.aff2axcodes(affine))   # Nifti file with LAS format.

path_bvecs = "/public2/lijialong/Data/real/xu_you_sheng/bvec_"+str(num_id)+"_sel" + dir_num + ".txt"
path_bvals = "/public2/lijialong/Data/real/xu_you_sheng/bval_"+str(num_id)+"_sel" + dir_num + ".txt"
bvecs = np.loadtxt(path_bvecs)
bvals = np.loadtxt(path_bvals)
bvecs = np.transpose(bvecs)
# ************************important***********************
# Used for data exported from dcm2niix, typically in the Nifti file with LAS format.
# Otherwise, you need to invert one or more axes of the gradient table and verify the modification using Color FA map.
bvecs[:, 0] = bvecs[:, 0] * -1.0       # LSA format data should invert the x-axis
gtab = gradient_table(bvals, bvecs)
bvals = bvals / 1000.0
bvecs_x = np.copy(bvecs)



# %% WLLS
dti_wls = dti.TensorModel(gtab, fit_method=fit_method, return_S0_hat=True)
tenfit = dti_wls.fit(data_mask)
n_s0 = tenfit.S0_hat * mask
a_evecs = tf.convert_to_tensor(tenfit.evecs)  # 1列1向量
a_evals = tf.convert_to_tensor(tenfit.evals)
a_evals = tf.linalg.diag(a_evals)
tensor_rc = tf.matmul(a_evecs, tf.matmul(a_evals, tf.transpose(a_evecs, perm=[0, 1, 2, 4, 3]))).numpy()
tensor_rc = np.reshape(tensor_rc, (tensor_rc.shape[0], tensor_rc.shape[1], tensor_rc.shape[2], 9))
n_tensor = tensor_rc[:, :, :, (0, 4, 8, 1, 5, 2)] * mask[..., np.newaxis]  # order[Dxx, Dyy, Dzz, Dxy, Dyz, Dxz]
n_tensor = n_tensor * 1000.0
n_fa, n_md, n_cfa, n_ad, n_rd = tso.tensor2metric(n_tensor, mask)



# %% DoDTI
print('** Normalization')
index_b0 = np.where(bvals == 0)
data_flatten = np.sort(data_mask[..., index_b0].flatten())
# ************************important***********************
# The scaling ratio should be adjusted, as it will affect the model's performance
index_max = round(len(data_flatten) * 0.995)  # Excluded outliers
pseudo_max = data_flatten[index_max]
noise = (data - np.min(data)) / (pseudo_max - np.min(data))
print('nf_max:%f, psudo_max:%f, min:%f' % (np.max(data), pseudo_max, np.min(data)))

# plt.figure()
# plt.hist(noise.flatten(), bins=100, histtype='step', range=(0.000001, 1.2))
# plt.savefig('./check/hist_dti')
# plt.show()
# plt.close()

# log_transform
# ************************important***********************
# Flip the image, as the clinical data in LAS format differs from the training data in LPS format.
n_dti_input = np.flip(noise, axis=1)      
mask_input = np.flip(mask, axis=1)
n_dti_input = np.maximum(n_dti_input, 1e-8)
n_dti_input = -1.0 * np.log(n_dti_input)
n_dti_input = n_dti_input[np.newaxis, ...]
print('n_dti_input.shape:', n_dti_input.shape)
bvals_test = np.broadcast_to(bvals, (n_dti_input.shape[0], bvals.shape[0]))
bvecs_test = np.broadcast_to(bvecs_x, (n_dti_input.shape[0], bvecs.shape[0], bvecs.shape[1]))
inputs = (n_dti_input, bvecs_test, bvals_test, mask_input[..., np.newaxis])

print('** DoDTI processing')
Nc = noise.shape[-1]               # input data channel
inpt_b_admmnet = tf.keras.layers.Input(shape=(None, None, None, Nc), name='input_dti')
inpt_bvec_admmnet = tf.keras.layers.Input(shape=(Nc, 3), name='bvecs')
inpt_bval_admmnet = tf.keras.layers.Input(shape=(Nc,), name='bvals')
inpt_mask_admmnet = tf.keras.layers.Input(shape=(None, None, None, 1), name='mask')
para_map_admmnet = net_admmnet([inpt_b_admmnet, inpt_bvec_admmnet, inpt_bval_admmnet, inpt_mask_admmnet])
model_admmnet = tf.keras.Model(inputs=[inpt_b_admmnet, inpt_bvec_admmnet, inpt_bval_admmnet,
                                    inpt_mask_admmnet], outputs=para_map_admmnet, name='ADMMNet')
model_admmnet.load_weights(filepath=model_dir_admmnet)

maps_all_stage = model_admmnet(inputs)
pred_maps = np.squeeze(maps_all_stage[-1])
pred_maps = np.flip(pred_maps, axis=1)
pred_s0 = np.exp(-1.0 * pred_maps[..., 0]) * mask  # s0
pred_tensor = pred_maps[..., 1:] * mask[..., np.newaxis]  # tensor1k
pred_map = np.concatenate([pred_s0[..., np.newaxis], pred_tensor], axis=-1)  # s0 tensor1k
pred_dti = func.recon_dti_test(pred_map, bvecs_x, bvals)  # need to input s0 and tensor*1000, b_value/1000
pred_dti = np.array(pred_dti) * mask[..., np.newaxis]
pred_fa, pred_md, pred_cfa, pred_ad, pred_rd = tso.tensor2metric(pred_tensor, mask)

tensor_mrtrix = np.split(pred_tensor, [4, 5, 6], axis=-1)
tensor_mrtrix = np.concatenate([tensor_mrtrix[0], tensor_mrtrix[2], tensor_mrtrix[1]], axis=-1)
save_nifti(save_img_path + 'dodti_'+str(num_id)+'_sel' + dir_num + '_tensor_MRtrix.nii.gz', tensor_mrtrix * 0.001, affine)  # Dxx Dyy Dzz Dxy Dxz Dyz
save_nifti(save_img_path + 'dodti_'+str(num_id)+'_sel' + dir_num + '.nii.gz', pred_dti, affine)
save_nifti(save_img_path + 'dodti_'+str(num_id)+'_sel' + dir_num + '_fa.nii.gz', pred_fa, affine)
save_nifti(save_img_path + 'dodti_'+str(num_id)+'_sel' + dir_num + '_md.nii.gz', pred_md, affine)



# %%figure show and save
plt.figure(dpi=600, figsize=(2, 5))
plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
plt.subplot(521), plt.imshow(np.rot90(n_fa[..., slice], k=1), cmap='gray', vmin=0, vmax=1), plt.axis('off')
plt.subplot(522), plt.imshow(np.rot90(pred_fa[..., slice], k=1), cmap='gray', vmin=0, vmax=1), plt.axis('off')
plt.subplot(523), plt.imshow(np.rot90(n_md[..., slice], k=1), cmap='gray', vmin=0, vmax=3), plt.axis('off')
plt.subplot(524), plt.imshow(np.rot90(pred_md[..., slice], k=1), cmap='gray', vmin=0, vmax=3), plt.axis('off')
plt.subplot(525), plt.imshow(np.rot90(n_ad[..., slice], k=1), cmap='gray', vmin=0, vmax=3), plt.axis('off')
plt.subplot(526), plt.imshow(np.rot90(pred_ad[..., slice], k=1), cmap='gray', vmin=0, vmax=3), plt.axis('off')
plt.subplot(527), plt.imshow(np.rot90(n_rd[..., slice], k=1), cmap='gray', vmin=0, vmax=3), plt.axis('off')
plt.subplot(528), plt.imshow(np.rot90(pred_rd[..., slice], k=1), cmap='gray', vmin=0, vmax=3), plt.axis('off')
plt.subplot(529), plt.imshow(np.rot90(n_cfa[..., slice, :], k=1), cmap='gray', vmin=0, vmax=0.2), plt.axis('off')
plt.subplot(5,2,10), plt.imshow(np.rot90(pred_cfa[..., slice, :], k=1), cmap='gray', vmin=0, vmax=0.2), plt.axis('off')
plt.savefig(
    os.path.join(save_img_path, 'xu_'+str(num_id)+'_sel' + dir_num + '_slice' + str(slice) + '_Metrics.png'))
plt.show()
plt.close()
