# 3D atherosclerotic plaque distribution and composition
Plaque analysis in 3D aorta imaging

### Plaque prediction

The script to train/fine-tune a model is `plaque_prediction/train_unet.py`. It takes as input an npz file containing training and validation set in the following sequence: train images, train labels, validation images, validation labels. The images should be normalized and the dimensions of images/labels are number_of_images, width, height, channels. We used (x,608,480,1). To test the script the data set provided in data/plaque_data_small.npz can be used (since this is only an example change lines 125 and 126 to arr_0 and arr_1 in train_unet.py).

The notebook `plaque_prediction/predict_plaque.ipynb` shows how to use a trained model to predict plaque.

### Anatomy segmentation

The script to train the 3D segmentation model is provided in `anatomy_segmentation/train_unet_3d.py`. The required input is a directory containing the training examples (aortas) and atlases of anatomy labels. Aortas and atlases are saved in the dimensions 300x260x190 pixels in nii.gz format (to process with sitk). Aorta lumen is filled in and the whole aorta is represented as binary image. Atlases contain labels 0-6 (0=background, 1=desc, 2=arch, 3=lsa, 4=lcca, 5=bca, 6=other). An example of an aorta and atlas are given in the data folder to test the code.

The notebook `anatomy_segmentation/anatomy_prediction.ipynb` gives an example on how to use a trained network to predict the anatomical structures in the aorta. It also shows the critical step of how to post-process the network predictions.

The weights of a trained 3D U-net are available [here](https://drive.google.com/file/d/1we2AUBz2tsk52HQd29v6Ue1CZEe8eGEi/view?usp=sharing)  
### Used library versions
* python 3.6.8
* numpy 1.16.3
* tensorflow 1.13.1
* keras 2.2.4
* pillow 6.0.0
* SimpleITK 1.2.4
* scikit-image 0.17.dev0
* scipy 1.2.1
* Augmentor 0.2.3

The U-net implementation is inspired by [this repository](https://github.com/shibuiwilliam/Keras_Autoencoder).

When using this code, please cite: [https://www.nature.com/articles/s41598-020-78632-4](https://www.nature.com/articles/s41598-020-78632-4)
