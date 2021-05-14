# DATA AUGMENTATION:
ALBUMENTATIONS library is used for image augmentation,which applies a no. of image transformations like Horizontal Flip, Vertical Flip, Rotate, etc.

# MASKING IMAGES:
Segimages contains masked images which contains 4 classes. But here we are about to do single class segmentation so left ventricle is alone segmented using inrange() function of CV2 library.

# MODEL 
- PRETRAINED MODEL HAS BEEN INCLUDED TO DIRECTLY TEST THE IMAGES - https://drive.google.com/file/d/1QDCsum1VqDyj3PZUrkdy4-daZ_M-Pu_X/view?usp=sharing

# TESTING :
Model is tested with images from the given dataset that have not been used for training. Those Images are included above. 

For verifying robustness of the model, a random Short axis Cardiac MRI image was taken from the 
internet and it was segmented. visual results appear to be appealing.
<p align = "center">
<img src = "https://user-images.githubusercontent.com/72727518/117295993-28ecd500-ae92-11eb-8f03-38b07ada39fa.png" width = "500" height = "300">
</p>
With further more of training with various datasets and parameter optimisation the model can be made even accurate.

# USAGE :
- Download the dataset and the model file to ur local computer and paste the test.py file in the same folder and run the file.
- If u wish to modify the architecture or optimize the parameters and train it use the colab file single_class_segmentation.ipynb 
