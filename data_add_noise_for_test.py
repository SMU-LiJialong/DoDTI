#!usr/bin/env python
# encoding: utf-8

#****************************
# Add noise for noise-free simulated testing data.
# In our paper, the simulated testing data consists of 11 subjects.
# However, the testing data is not included in this demo.
# Therefore, you will need to modify this code accordingly.
#***************************

import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
noise_type = 'rician'
# noise_type = 'guassian'
noise_level = np.linspace(0.01, 0.04, 4)  # Four noise levels were applied to each dataset
# The training data indices range from 01 to 20, while the testing data indices range from 21 to 32.
# index = np.linspace(21, 32, 11, dtype='int')  
index = [1]   # for demo
num_out = 0
for i in index:
    for k in range(4):
        num_in = '%02d' % i
        load_path = './data/nf_norm/nf_norm_' + num_in + '.nii.gz'  # normalized data
        img = nib.load(load_path)
        img_affine = img.affine
        data_nf_norm = img.get_fdata().astype('float32')

        #%%adding noise according to noise type and sigma_p
        sigma_p = noise_level[k]
        nr = sigma_p * np.random.standard_normal(data_nf_norm.shape) + data_nf_norm
        if noise_type == 'guassian':
            data_n_norm = nr
            print('Adding guassian noise, sigma=', sigma_p)
        elif noise_type == 'rician':
            ni = sigma_p * np.random.standard_normal(data_nf_norm.shape)
            data_n_norm = np.sqrt(nr ** 2 + ni ** 2)
            print('Adding rician noise, sigma=', sigma_p)
        else:
            print('noise type error')

        #%% Check for noise 
        plt.figure(figsize=(4, 4))
        plt.subplots_adjust(left=0, right=1, wspace=0, hspace=0, bottom=0, top=1)
        plt.subplot(211)
        plt.imshow(np.hstack((data_nf_norm[56, :, :, 0], data_n_norm[56, :, :, 0])), vmin=0, vmax=0.4,
                   cmap='gray'), plt.axis('off')
        plt.subplot(212)
        plt.imshow(np.hstack([data_nf_norm[56, :, :, 1], data_n_norm[56, :, :, 1]]), vmin=0, vmax=0.2,
                   cmap='gray'), plt.axis('off')
        plt.savefig('./check/add_noise.png')
        plt.show()
        plt.close()

        #%% save data_n as nii.gz
        save_path_n = './data/rician_norm/rician' + num_in + '_sigma'+ str(int(sigma_p * 1000)) +'.nii.gz'  # rician
        nib.Nifti1Image(data_n_norm, img_affine).to_filename(save_path_n)
        print('Finish adding %s noise to data_%s with sigma=%f \n' % (noise_type, num_in, sigma_p))
print('ok')

