# DoDTI
The implement of the following paper: "Reliable Deep Diffusion Tensor Estimation: Rethinking the Power of Data-Driven Optimization Routine" [1]. 
(The manuscript of this project is being reviewed, once accepted we will release all the code)

## Architecture of the proposed DoDTI
![image](https://github.com/user-attachments/assets/9832aa5e-a4db-477c-a04c-6c8d0c05b7ef)  
Architecture of the proposed DoDTI. (a) The input includes at least six diffusion-encoding volumes plus one non-DW volume, and corresponding gradient directions and b values. The output includes the predicted non-DW volume and six tensor element maps. The n-th iteration of ADMM corresponds to the n-th stage of the deep neural network enclosed within the dashed box, in which the fitting block (X), the auxiliary variable block (Z), and the multiplier block (M) are sequentially updated. (b) The structure of the denoiser, which is embedded within the auxiliary variable block.

   
## Data generation pipeline

Please refer to the paper[1] for detailed information on the training data generation process.

## HCP data
The example dataset is provided by the Human Connectome Project [2]. Please acknowledge the source to the WU-Minn-Oxford HCP. The orginal data is available at [https://www.humanconnectome.org/study/hcp-young-adult](https://www.humanconnectome.org/study/hcp-young-adult/).
The example data was reconstructed using the pipeline mentioned in our paper[1].  

The reconstructed DW images are stored in the './data/nf_unnorm' folder (noise-free unnormalized).   

The diffusion tensor field is stored in the './data/tensor' folder, each volume is given by [Dxx Dyy Dzz Dxy Dyz Dxz]   

The mask file is stored in the './data/mask' folder.   

The bvalue and bvec files are stroed in the './data/gradient' folder.   
<br/><br/>

## data_add_noise_for_train.py
Normalize and add noise to the noise-free data, as the input for the network.
<br/><br/>

## data_combine_train_valid.py
Generate paired data sets for training and validation using noise-free DTI data, noisy DTI data, and diffusion tensor field.
<br/><br/><br/>

## data_add_noise_for_test.py
Generate simulated noisy data for testing.
<br/><br/>

## train.py
Train DoDTI.

We also provide a model trained on 20 sets of 6-direction data, which demonstrates good accuracy and generalization.    

However, due to the inherent dependency of deep learning on the training data, we recommend either retraining the model with your own data or using our model as a pre-trained model.  

The model is stroed at './exp1_mix/model/model0250.h5'.
<br/><br/>

## test_for_simulated_data.py
Test simulated data. Please do not use this file to process clinical data, as there are important details to be noted.
<br/><br/>

## test_for_clinical_data.py
Test clinical data.
<br/><br/>

## net_dodti.py
The network of DoDTI.  
<br/><br/>

## net_denoiser.py
The denoising network, which is embedded in DoDTI.
<br/><br/>

## functions_cal_metric.py, functions_tensor_eigen.py function_wlls_3D.py
   Some functions to be called by the newtork.
<br/><br/>

## Refereces
[1] Li, J., Zhang, Z., Chen, Y., Lu, Q., Wu, Y., Liu, X., Feng, Q., Feng, Y. and Zhang, X., 2024. Reliable Deep Diffusion Tensor Estimation: Rethinking the Power of Data-Driven Optimization Routine. arXiv preprint arXiv:2409.02492.  

[2] Fan, Q., Witzel, T., Nummenmaa, A., Van Dijk, K.R.A., Van Horn, J.D., Drews, M.K., Somerville, L.H., Sheridan, M.A., Santillana, R.M., Snyder, J., Hedden, T., Shaw, E.E., Hollinshead, M.O., Renvall, V., Zanzonico, R., Keil, B., Cauley, S., Polimeni, J.R., Tisdall, D., Buckner, R.L., Wedeen, V.J., Wald, L.L., Toga, A.W., Rosen, B.R., 2016. MGH-USC Human Connectome Project datasets with ultra-high b-value diffusion MRI. Neuroimage 124, 1108-1114.
