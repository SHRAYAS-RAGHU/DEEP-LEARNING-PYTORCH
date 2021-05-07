# DATA AUGMENTATION:
ALBUMENTATIONS library is used for image augmentation,which applies a no. of image transformations like Horizontal Flip, Vertical Flip, Rotate, etc.

# CREATING MASKS:
The given segimages directory contains mask images. But PYTORCH requires the target image to have a pixel value which denotes a class. For example if there are 4 classes each pixel must have either 0,1,2 & 3.

# TESTING :
Model is tested with images from the given dataset that have not been used for training.
With further more of training with various datasets and parameter optimisation the model can be made even accurate.

# USAGE :
- Download the dataset and put it in ur drive and import and unzip the dataset using the cells provided in the .ipynb file.
- If u wish to modify the architecture or optimize the parameters and train it use the colab file single_class_segmentation.ipynb 
