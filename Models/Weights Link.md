# Weights for trained CNN models - [Google Drive](https://drive.google.com/drive/folders/1HlIjfr5C9iWLHDGGwbOCWcOccSubkTWe?usp=sharing)

## _Results for various architectures_


### Initial testing for Pneumonia localisation
* Implemented GradCAM functionality to check Pneumonia localisation of baseline and pre-trained models.
  
#### TinyVGG `CNN_GradCAM_v1.pth`
* Test Loss: 0.305 | Test Acc: 88.28% | Sensitivity: 91.79% | Specificity: 81.62%
* **Problem:** Focus on unwanted features, extremely poor localisation of Pneumonia. Heavy overfitting.
  
  <img src="https://github.com/user-attachments/assets/296b7743-5c1d-4899-b955-6cf60a0b0f39" alt="Grid of GradCAMs" width="500" height="500"><br>
  <small> `Pneumonia Class | Good` here "Good" means prediction score >= 85% | `blank boxes` indicate wrong predictions <small>


#### VGG-16 `VGG-16_GradCAM_v2.pth`
* Test Loss: 0.362 | Test Acc: 83.91% | Sensitivity: 88.72% | Specificity: 75.07%
* **Progress:** Improved but unsatisfactory localisation.

  <img src="https://github.com/user-attachments/assets/bae4d462-d7b7-4928-90c8-8ebc4b8eb06b" alt="Grid of GradCAMs" width="500" height="500"><br>


#### ResNet-101 `ResNet-101_GradCAM_v1.pth`
* Test loss: 0.369 | Test acc: 83.59% | Sensitivity: 95.38% | Specificity: 62.82%
* **Progress:** Improved localisation after adding an additional FC layer with 10k params.

  <img src="https://github.com/user-attachments/assets/4bd31cb7-2e10-4460-88f7-df34211617ba" alt="Grid of GradCAMs" width="500" height="500"><br>


#### ResNet-152 `ResNet-152_GradCAM_v1.pth`
* Test Loss: 0.379 | Test acc: 84.69% | Sensitivity: 93.33% | Specificity: 70.09%
* **Progress:** Similar/worse localisation compared to ResNet-101 with similar evaluation metrics.

  <img src="https://github.com/user-attachments/assets/9f695585-faa6-4294-ae72-aadb2cb1e63b" alt="Grid of GradCAMs" width="500" height="500"><br>

_Finally,_

#### BEST OVERALL PERFORMANCE AND LOCALISATION: ResNet-101 `ResNet-101_GradCAM_v1.pth`
* ResNet-101 is now re-trained and optimized by increasing the additional FC layer to have 220k params and by slowly un-freezing ResNet_layer4.
* However, there is a noticeable focus on void regions surrounding the skeleton. To fix this, a small center-crop was added to training images before feeding them to the model, thus preventing the model from "learning" these void regions.

   <img src="https://github.com/user-attachments/assets/d80871ad-91a9-46a5-befe-5d34128d6119" alt="Grid of GradCAMs" width="800" height="400"><br>

* Finally, ~17M parameters were re-trained (~38% of ResNet-101) and optimized to achieve the current "best" model: `ResNet-101_GradCAM_Cropped_v4.pth`
* Test Loss: | Test acc: | Sensitivity: | Specificity:
* **Progress:** Obtained noticeable improvement in Pneumonia localisation and evaluation metrics. The model is now ready to create an enhanced data set.

  <img src="" width="500" height="500"><br>

*model architectures in PyTorch can be found in the GradCAM.ipynb notebook*
