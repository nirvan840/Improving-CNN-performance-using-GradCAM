# Improving CNN performance using GradCAM [(Collab Link)](https://colab.research.google.com/drive/1ffbipSZC2AevoZYl6MS8TJqXMXNHXRvP?usp=sharing)


### _Motivtion_
**Doubt:** Unexpected high performance of shallow CNN (4 Conv2d layers only) on Pneumonia Dataset. <br>
**Hypothesis:** Possible model overfitting on unwanted features rather than Pneumonia. 

### _Problem Statement_
Improving Convolutional Neural Network classification and feature localisation for Pneumonia Chest X-rays in the scenario of lack of extensive
annotated data and access to extensive GPU Training architectures.

### _Approach_

![image](https://github.com/user-attachments/assets/e14a62a0-4af0-40e6-94ff-8a52712a726d)
<small> _A model-independent, self-sufficient, cyclic process is developed to achieve better Pneumonia localisation and improve performance metrics on the given dataset._ <small>

------------------------------------------------------------------------------------------------------------------------------------------------------------------------
&nbsp;

## _INSTRUCTIONS_

### General
* **Code**: **Pytorch** | **Data-set**: [paultimothymooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
* After opening the notebook in Collab, please go to `File > Save as copy in Google Drive` to experiment with the code after reading the **Data Handing** section below.
  
### Data Handling 
* **.zip file** containing the **Pneumonia Dataset** must be uploaded to the `My Drive` folder of the **Google Drive** mounted to the collab notebook.
* .zip file is** automatically extracted** to `drive\My Drive\ML Data Sets` while **removing any corrupted images**.
* To use **external data-sets**, place them in `drive\My Drive\ML Data Sets` and change variable names.

### CNN Models 
* Baseline model architecture: [TinyVGG](https://poloclub.github.io/cnn-explainer/) 
* Links to **.pth weights** of the trained models can be found in the `models` folder.
* **popular pre-trained architectures** (VGG-16, VGG-19, ResNet-50, ResNet-101, ResNet-152) can be used and corresponding models loaded.
* Model, optimizer & scheduler **state_dicts** can be saved to Google Drive (`drive\MyDrive\ML Models\<xyz.pth>`). These can be loaded in future for seamless resumption of progress.<br>

_mismatch in model parameters while loading models can be a result of modified architecture in the code_

------------------------------------------------------------------------------------------------------------------------------------------------------------------------
&nbsp;

## _Results for various architectures_

### Initial testing for Pneumonia localisation
* Implemented GradCAM functionality to check Pneumonia localisation of baseline and pre-trained models.
* Concluded the optimal model architecture and optimized it. 

### Different Architectures
##### TinyVGG `CNN_GradCAM_v1.pth`
* Test Loss: 0.305 | Test Acc: 88.28% | Sensitivity: 91.79% | Specificity: 81.62%
* **Problem:** Focus on unwanted features, extremely poor localisation of Pneumonia. Heavy overfitting.
  
  <img src="https://github.com/user-attachments/assets/296b7743-5c1d-4899-b955-6cf60a0b0f39" alt="Grid of GradCAMs" width="500" height="500"><br>
  <small> `Pneumonia Class | Good` here "Good" means prediction score >= 85% <br> `blank boxes` indicate wrong predictions <small>


##### VGG-16 `VGG-16_GradCAM_v2.pth`
* Test Loss: 0.362 | Test Acc: 83.91% | Sensitivity: 88.72% | Specificity: 75.07%
* **Progress:** Improved but unsatisfactory localisation.

  <img src="https://github.com/user-attachments/assets/bae4d462-d7b7-4928-90c8-8ebc4b8eb06b" alt="Grid of GradCAMs" width="500" height="500"><br>


##### ResNet-101 `ResNet-101_GradCAM_v1.pth`
* Test loss: 0.369 | Test acc: 83.59% | Sensitivity: 95.38% | Specificity: 62.82%
* **Progress:** Improved localisation after adding FC layer with 10k params.

  <img src="https://github.com/user-attachments/assets/4bd31cb7-2e10-4460-88f7-df34211617ba" alt="Grid of GradCAMs" width="500" height="500"><br>


##### ResNet-152 `ResNet-152_GradCAM_v1.pth`
* Test Loss: 0.379 | Test acc: 84.69% | Sensitivity: 93.33% | Specificity: 70.09%
* **Progress:** Similar/worse localisation compared to ResNet-101 with similar evaluation metrics.

  <img src="https://github.com/user-attachments/assets/9f695585-faa6-4294-ae72-aadb2cb1e63b" alt="Grid of GradCAMs" width="500" height="500"><be>


### Best Overall Localisation and Performance: ResNet-101 `ResNet-101_GradCAM_v1.pth`
* ResNet-101 is re-trained and optimized by increasing the additional FC layer to have 220k params and by slowly un-freezing ResNet_layer4.
* However, there is an unwated focus on **void regions** surrounding the skeleton. To fix this, a small center-crop was added to training images before feeding them to the model, thus preventing the model from "learning" these void regions.

   <img src="https://github.com/user-attachments/assets/d80871ad-91a9-46a5-befe-5d34128d6119" alt="Grid of GradCAMs" width="800" height="400"><br>

* Finally, ~17M parameters were re-trained (~38% of ResNet-101) and optimized to achieve the current "best" model: `ResNet-101_GradCAM_Cropped_v4.pth`
* Test Loss: | Test acc: | Sensitivity: | Specificity: ### COMPLETE ###
* **Progress:** Obtained **noticeable improvement** in Pneumonia localisation and evaluation metrics. The model is now ready to create an enhanced data set.

  <img src="" width="500" height="500"><br> ### COMPLETE ###

------------------------------------------------------------------------------------------------------------------------------------------------------------------------
&nbsp;

## _ENHANCED DATASET RESULTS_

### Enhanced Data Set
* Train Data enhanced by overlaying accurate GarCAMs.
  * "**enhanced**" - important features emphasized while the rest of the image is suppressed.
  * "**accurate**" - the assumption is that GradCAMs localized to the lung are fairly accurate.
  
  <img src="" width="500" height="500"><br> ### COMPLETE ###

### Results using Enhanced Data Set
* Classifier layer of `ResNet-101_GradCAM_Cropped_v4.pth`, 220k params, retrained.
* Test Loss: | Test Acc: | Sensitivity: | Specificity: ### COMPLETE ###

  Improved GradCAMs<br>
  <img src="" width="500" height="500"><br> ### COMPLETE ###

------------------------------------------------------------------------------------------------------------------------------------------------------------------------
&nbsp;

## _CONCLUSION_

*

------------------------------------------------------------------------------------------------------------------------------------------------------------------------
&nbsp;

## _REFERENCES_
* GradCAM - [JimEverest](https://github.com/JimEverest/CAM)
* Lung Segmentation - [IlliaOvcharenko's repo](https://github.com/IlliaOvcharenko/lung-segmentation/tree/master?tab=readme-ov-file)

