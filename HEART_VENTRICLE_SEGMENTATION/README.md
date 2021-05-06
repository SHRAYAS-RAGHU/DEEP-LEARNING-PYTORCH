# BINARY SEGMENTATION:

Each pixel value is classified into one of the two classes (Background, Left Ventricle). **2D-UNET** model is used for the task. **Sigmoid** activation is applied to the last layer and BCE loss is used for backpropagation and training.

# UNET MODEL:
![image](https://user-images.githubusercontent.com/72727518/117279604-7f044d00-ae7f-11eb-90d2-c2809bdaebad.png))

# DATASET :

FOR THE DATASET DRIVE LINK HAS BEEN SHARED

## MRIIMAGES 
- CONTAINS ORIGINAL IMAGE TO BE FED INTO THE UNET MODEL - https://drive.google.com/file/d/1EOu3xcZz2_bp3ZnNwRsJ0WuIHmYJwmqQ/view?usp=sharing

## SEGIMAGES 
- CONTAINS MASK OF IMAGES THAT IS TO BE USED AS TARGET - https://drive.google.com/file/d/1-5Nd_mCDRRqTFqmAUOtWapi6NSFtgKR9/view?usp=sharing

## MODEL 
- PRETRAINED MODEL HAS BEEN INCLUDED TO DIRECTLY TEST THE IMAGES - https://drive.google.com/file/d/1QDCsum1VqDyj3PZUrkdy4-daZ_M-Pu_X/view?usp=sharing

# TESTING :
Model is tested with images from the given dataset that have not been used for training. For robustness of the model, a random Short axis Cardiac MRI image was taken from the internet and it was segmented. visual results appear to be appealing. with further more of training with various datasets and parameter optimisation the model can be made even accurate.
