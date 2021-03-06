# SEGMENTATION:

Each pixel value is classified into one of the two or many classes. **2D-UNET** model is used for the task.

## BINARY SEGMENTATION:

Each pixel value is classified into one of the two classes (Background and left-ventricle).
**Sigmoid** activation is applied to the last layer and **BCE loss** is used for backpropagation and training.

## MULTI-CLASS SEGMENTATION:

Each pixel value is classified into one of the four classes (Background, right-ventricle, myocardium and left-ventricle).
**Log_Softmax** activation is applied to the last layer and **NLL loss** is used for backpropagation and training.
PYTORCH has **Cross-Entropy loss** function that applies **Log_Softmax** activation and **NLL loss**.

# UNET MODEL:
<p align = "center">
<img src = "https://user-images.githubusercontent.com/72727518/130623287-032ad9de-bb91-4fe4-a376-adcc9c17f094.jpg">
</p>
                                                                                                                 
<p align = "center">
<img src = "https://user-images.githubusercontent.com/72727518/130623913-36cc5c7a-7078-47b5-9b47-570b1060fb2f.png">
</p>                                                                                                              
# DATASET :

Simplified ACDC Short axis cardiac MRI is used as dataset.

FOR THE DATASET DRIVE LINK HAS BEEN SHARED

## MRIIMAGES 
- CONTAINS ORIGINAL IMAGE TO BE FED INTO THE UNET MODEL - https://drive.google.com/file/d/1EOu3xcZz2_bp3ZnNwRsJ0WuIHmYJwmqQ/view?usp=sharing

## SEGIMAGES 
- CONTAINS MASK OF IMAGES THAT IS TO BE USED AS TARGET - https://drive.google.com/file/d/1-5Nd_mCDRRqTFqmAUOtWapi6NSFtgKR9/view?usp=sharing

# CREDITS : 
Aladdin Persson - https://youtu.be/IHq1t7NxS8k
