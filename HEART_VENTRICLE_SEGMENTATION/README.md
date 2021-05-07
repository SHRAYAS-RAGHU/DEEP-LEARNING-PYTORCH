# BINARY SEGMENTATION:

Each pixel value is classified into one of the two classes (Background, Left Ventricle). **2D-UNET** model is used for the task. **Sigmoid** activation is applied to the last layer and BCE loss is used for backpropagation and training.

# UNET MODEL:
![image](https://user-images.githubusercontent.com/72727518/117279604-7f044d00-ae7f-11eb-90d2-c2809bdaebad.png))

# DATASET :

Simplified ACDC Short axis cardiac MRI is used as dataset.

FOR THE DATASET DRIVE LINK HAS BEEN SHARED

## MRIIMAGES 
- CONTAINS ORIGINAL IMAGE TO BE FED INTO THE UNET MODEL - https://drive.google.com/file/d/1EOu3xcZz2_bp3ZnNwRsJ0WuIHmYJwmqQ/view?usp=sharing

## SEGIMAGES 
- CONTAINS MASK OF IMAGES THAT IS TO BE USED AS TARGET - https://drive.google.com/file/d/1-5Nd_mCDRRqTFqmAUOtWapi6NSFtgKR9/view?usp=sharing

## MODEL 
- PRETRAINED MODEL HAS BEEN INCLUDED TO DIRECTLY TEST THE IMAGES - https://drive.google.com/file/d/1QDCsum1VqDyj3PZUrkdy4-daZ_M-Pu_X/view?usp=sharing

# TESTING :
Model is tested with images from the given dataset that have not been used for training. Those Images are included above. 

For verifying robustness of the model, a random Short axis Cardiac MRI image was taken from the 
internet and it was segmented. visual results appear to be appealing.

![image](https://user-images.githubusercontent.com/72727518/117295993-28ecd500-ae92-11eb-8f03-38b07ada39fa.png)

With further more of training with various datasets and parameter optimisation the model can be made even accurate.

# USAGE :
- Download the dataset and the model file to ur local computer and paste the test.py file in the same folder and run the file.
- If u wish to modify the architecture or optimize the parameters and train it use the colab file single_class_segmentation.ipynb 

# CREDITS : 
Aladdin Pearson - UNET MODEL PAPER TO CODE
