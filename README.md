# Introduction
### This repository contains some parts of code used in the paper: "Covid-19 detection from chest X-Ray images using deep convolutional neural networks". This code was executed in Google Colab taking advantage of the GPU access provided by Google.  This work is based on the use of convolutional neural networks. The experimental results of our proposed model show a high accuracy of 97.33%, sensitivity of 98% and specificity of 100% when detecting covid-19 from chest X-ray images of covid-19 positive cases, normal case images and viral pneumonia case images. Also shared links to download the data and models used during the training and testing phases are provided. 

# Used data
Referring to the article, the images used in this work have been downloaded from several sources. After the pre-processing operations (elimination of similar images, offline data augmentation), these images will be used for training and testing of the different models are available in Google Drive via the folder [covid19_data](https://drive.google.com/drive/folders/1DNsSVVV4sYOcONsOQXWXmB2l5XUvyAaM?usp=sharing). 

The notebook preparing_datasets.ipynb allows to put the image data centered between [-1,1] and their labels in nupmpy tables as mentioned in the following table: 

| Data                | Downloads                     |
|---------------------|-------------------------------|
| Data for training   | [data_train.npy](https://drive.google.com/file/d/1w7yB602LJ273sxWmbGaCn5FeK4X4083W/view?usp=sharing)  &&  [labels_train.npy](https://drive.google.com/file/d/1GLyB1v63zE4D9N4YXzpaPXeezNzJUxW-/view?usp=sharing) |
| Data for validation | [data_val.npy](https://drive.google.com/file/d/1-2lAPVJIxD1l5OyF5vBcycfbAZ_BSyYn/view?usp=sharing)  &&    [labels_val.npy](https://drive.google.com/file/d/1-4ncP6GUWSJPDNkmKW3APJxzuZTTQpIK/view?usp=sharing)   |

# Training models
The notebook covid-19_Training_Models.ipynb allows to train 4 models based on VGG16, VGG19, InceptionV3 and ResNet50 via transfer learning for automatic detection of covid-19 from X-ray chest images. The models obtained with are available on Google Drive via the following links: 

| Model based on | Validation Accuracy(%) | Downloads     |
|----------------|------------------------|---------------|
| VGG16          |        97,67           | [Download](https://drive.google.com/file/d/1-2-g8KpSMUxNHKhkZwPTOHy6po-jiWMg/view?usp=sharing) |
| InceptionV3    |        97,33           | [Download](https://drive.google.com/file/d/1-ioN0DvOijsvYQUY10clDCmzjO9whSgd/view?usp=sharing) |
| VGG19          |        97,00           | [Download](https://drive.google.com/file/d/1Y1Gey2BOqThZJThKo5S3vy13QIc6vfw0/view?usp=sharing) |
| ResNet50       |        96,33           | [Download](https://drive.google.com/file/d/1-BBC9uQ8Ji8qI1L-_bxpG2GvE9S8BNRu/view?usp=sharing) |      

# Testing models
The notebook covid-19_Testing_Models.ipynb allows the evaluation of the 4 models obtained, providing their confusion matrices and the different metrics such as sensitivity, specificity, precision, etc. Also the code of this notebook allows the plotting of the ROC (Receiver Operating Characteristics) curves of the different models. 
