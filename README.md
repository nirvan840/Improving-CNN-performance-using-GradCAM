# Improving CNN performance using GradCAM [(Collab)](https://colab.research.google.com/drive/1ffbipSZC2AevoZYl6MS8TJqXMXNHXRvP?usp=sharing)


### _Motivtion_
_**Doubt:**_ Unexpected high performance of shallow CNN (4 Conv2d layers only) on Pneumonia Dataset. <br>
_**Hypothesis:**_ Possible model overfitting on unwanted features rather than Pneumonia. 

### _Problem Statement_
Improving Convolutional Neural Network classification and feature localisation for Pneumonia Chest X-rays in the scenario of lack of extensive
annotated data and access to extensive GPU Training architectures.

### _Approach_

* A model-independent, self-sufficient, **cyclic process** is developed to achieve better Pneumonia localisation and improve performance metrics on the given dataset.
* The **recursive optimization cycle for CNN models**: GradCAM generation → Enhanced data-set construction by overlaying GradCAMs on input → U-Net Lung Segmentation → Model tuning on enhanced data-set.
* The proposed cycle **tackles self-imposed challenges of limited data** and suboptimal localization by **iteratively augmenting and re-utilizing the original dataset**, thereby refining the model’s focus on pneumonia-specific features.
  
  ![image](https://github.com/user-attachments/assets/e14a62a0-4af0-40e6-94ff-8a52712a726d)
  _Image showing one aforementioned cycle_

&nbsp;

## _INSTRUCTIONS_

<details>
  <summary> <b><i>Details</i></b> </summary>
  
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
  
</details>

------------------------------------------------------------------------------------------------------------------------------------------------------------------------
&nbsp;

## _RESULTS FOR VARIOUS ARCHITECTURES_

<details>
  <summary> <b><i>Details</i></b> </summary>
  
  ### Initial testing for Pneumonia localisation
  * Implemented GradCAM functionality to check Pneumonia localisation of baseline and pre-trained models.
  * Concluded the optimal model architecture and optimized it. 
  
  ### Different Architectures
  #### TinyVGG `CNN_GradCAM_v1.pth`
  * Test Loss: 0.305 | Test Acc: 88.28% | Sensitivity: 91.79% | Specificity: 81.62%
  * **Problem:** Focus on unwanted features, extremely poor localisation of Pneumonia. Heavy overfitting.
    
    <img src="https://github.com/user-attachments/assets/296b7743-5c1d-4899-b955-6cf60a0b0f39" alt="Grid of GradCAMs" width="400" height="400"><br>
    <small> `Pneumonia Class | Good` here "Good" means prediction score >= 85% <br> `blank boxes` indicate wrong predictions <small>
  
  
  #### VGG-16 `VGG-16_GradCAM_v2.pth`
  * Test Loss: 0.362 | Test Acc: 83.91% | Sensitivity: 88.72% | Specificity: 75.07%
  * **Progress:** Improved but unsatisfactory localisation.
  
    <img src="https://github.com/user-attachments/assets/bae4d462-d7b7-4928-90c8-8ebc4b8eb06b" alt="Grid of GradCAMs" width="400" height="400"><br>
  
  
  #### ResNet-101 `ResNet-101_GradCAM_v1.pth`
  * Test loss: 0.369 | Test acc: 83.59% | Sensitivity: 95.38% | Specificity: 62.82%
  * **Progress:** Improved localisation after adding FC layer with 10k params.
  
    <img src="https://github.com/user-attachments/assets/4bd31cb7-2e10-4460-88f7-df34211617ba" alt="Grid of GradCAMs" width="400" height="400"><br>
  
  
  #### ResNet-152 `ResNet-152_GradCAM_v1.pth`
  * Test Loss: 0.379 | Test acc: 84.69% | Sensitivity: 93.33% | Specificity: 70.09%
  * **Progress:** Similar/worse localisation compared to ResNet-101 with similar evaluation metrics.
  
    <img src="https://github.com/user-attachments/assets/9f695585-faa6-4294-ae72-aadb2cb1e63b" alt="Grid of GradCAMs" width="400" height="400"><be>
  
  
  ### Best Overall Localisation and Performance: ResNet-101 `ResNet-101_GradCAM_v1.pth`
  * There is an unwanted focus on **void regions** surrounding the skeleton.
  * To fix this, a small center-crop was added to training images before feeding them to the model, thus preventing the model from "learning" these void regions.
  
     <img src="https://github.com/user-attachments/assets/d80871ad-91a9-46a5-befe-5d34128d6119" alt="Grid of GradCAMs" width="650" height="320"><br>
  
  ### Optimizing ResNet-101_GradCAM_v1.pth
  * Finally, **~17M parameters** were re-trained** from scratch **(~38% of ResNet-101) using above** cropped train data** to achieve the current "best" model: `ResNet-101_GradCAM_Cropped_v4.pth`
  * `Test Loss: 0.264 | Test Acc: 91.56% | Sensitivity: 94.62% | Specificity: 85.90%`
  * **Progress:** Obtained **noticeable improvement** in Pneumonia localisation and evaluation metrics.<br> The model is now ready to create an enhanced data set.
  
    <img src="https://github.com/user-attachments/assets/5eaf8e0e-4665-4672-b4b3-be9d57656653" width="500" height="500"><br>
    <small> _GradCAMs showing improved localisation_ <small>

</details>

------------------------------------------------------------------------------------------------------------------------------------------------------------------------
&nbsp;

## _ENHANCED DATASET RESULTS_

<details>
  <summary> <b><i>Details</i></b> </summary>
  
  ### Enhanced Data Set
  * Train Data enhanced by overlaying accurate GarCAMs.
    * "**enhanced**" - important features emphasized (in "white") while the rest of the image is suppressed.
    * "**accurate**" - the assumption is that GradCAMs localized to the lung are fairly accurate.
    
      <img src="https://github.com/user-attachments/assets/c672799b-1b9c-4c76-8fca-2d5480763dd1" width="500" height="500"><br>
      _Image showing 49 images from the enhanced train data<br>post data transformations_
  
  ### Results using Enhanced Data Set `ResNet-101_GradCAM_Cropped_v4_enh_v1.pth`
* Re-trained previously "un-frozen" ~17M **ResNet-101_GradCAM_Cropped_v4.pth** params using above **enhanced data set**.
  * One cycle resulted in a **~15% improvement in the localization** of pneumonia, with only a **~3% decrease in test accuracy** with respect to **ResNet-101_GradCAM_Cropped_v4.pth.**
  * `Test Loss: 0.397 | Test acc: 88.59% | Sensitivity: 86.41% | Specificity: 91.88%`
  
    <img src="https://github.com/user-attachments/assets/bc2bef1c-8bb1-4c21-830e-086f981974e7" width="500" height="500"><br>
     _Improved GradCAMs post fine-tuning using enhanced data_

</details>
  
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
&nbsp;

## _CONCLUSION_

<details>
  <summary> <b><i>Details</i></b> </summary>

  ### Summary
  * We successfully can develop a **workaround for the limited data constraint** to iteratively obtain better localisation & identification of pneumonia.
  * During this process, as expected, test accuracy takes a small hit as the model diverges away from unwanted "easy" to detect features.
  
  ### Results on annotated images
  * <small> _Model was not trained on annotated data; the presence of arrows pointing to Pneumonia in the X-rays makes no difference in the model choice_ <small>
  
  * **Covid-19 Pneumonia X-ray**<br>
    <img src="https://github.com/user-attachments/assets/66eaa0d6-0a0c-4f6a-b26b-073bdcbc6a86" width="300" height="150"><br>
  
  * **Pneumonia X-ray**<br>
    <img src="https://github.com/user-attachments/assets/3e8939b2-fb09-46f7-bbab-2df2d8054985" width="300" height="150"><br>
  
  * **Pneumonia X-ray**<br>
    <img src="https://github.com/user-attachments/assets/81da8dee-e23a-4f3a-8a92-453165988912" width="300" height="150"><br>

### Post and Pre-Enhanced Data Fine-Tuning Comparison
  * Post-fine-tuning the model on enhanced data, we observe significantly better localisation of Pneumonia to the lungs.
  * The model also relies less on unwanted features and void regions while predicting.
  * The results **are limited to only one cycle** shown in the schematic above; further process iterations are bound to yield gradual but better results.

    <img src="https://github.com/user-attachments/assets/f546ab98-4333-4946-ba06-814d7d063906" width="3500" height="400"><br>
     _Final Comparison on the same 100 test images_

</details>

------------------------------------------------------------------------------------------------------------------------------------------------------------------------
&nbsp;

## _REFERENCES_
* <i>  GradCAM - [JimEverest](https://github.com/JimEverest/CAM)  </i>
* <i> Lung Segmentation - [IlliaOvcharenko's repo](https://github.com/IlliaOvcharenko/lung-segmentation/tree/master?tab=readme-ov-file) </i>

