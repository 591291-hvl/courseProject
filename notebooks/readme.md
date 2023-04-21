# Course prosject DAT255

## fastMONAI for medical image analysis

### Goals
- Explore the world of medical imaging. Gain hands-on experience solving core tasks using
deep learning.
- Get to know the fastMONAI library and other powerful deep learning frameworks for medical
imaging.
---

## This project contains

- [Datasets used](#datasets-used)
- [Loading data](#loading-data)
    - [Decathlon](#medical-segmentation-decathlon)
    - [MedNIST](#medmnist)
- [View data](#view-data)
- [Segmentation](#segmentation)
- [Classification](#classification)

---

## Datasets used

[Medical Segmentation Decathlon](http://medicaldecathlon.com/)

- Brain Tumours

Challenge: Complex and heterogeneously-located targets.

- Prostate

Challenge: Segmenting two adjoint regions with large inter-subject variations.

[MedNIST](https://medmnist.com/)

- Intercranial Aneurysm

Challenge: Unbalanced dataset.


---
## Loading data

- ### Medical Segmentation Decathlon

Decathlon datasets have a libary which makes it easy to download, which is done as below.
![decathlon-download](/notebooks/images/readmeImages/downloadDecathlon.png)

- ### Medmnist

MedNIST dataset also have a libary which makes it easy to download, shown below. But the file format is more challenging.

![MedNIST-download](/notebooks/images/readmeImages/downloadMedNIST.png)

This gives a .npz file which is a dictonary containing multiple keys of matrise lists.

![MedNIST-keys](/notebooks/images/readmeImages/MedNISTkeys.png)

Loading this filetype into any type of dataloader does not work as it need individual image paths(?). Therefore the only way i found to load the images was to create .nii.gz files of every image. Function that does this is shown below.

![MedNIST-CreateData](/notebooks/images/readmeImages/MedNISTCreateData.png)

This creates a new folder with a .nii.gz file for every entry in train/val/test, and a pandas dataframe containing filepath and corresponding label. This dataframe is usable for the MedDataset() loader.

---
## View data

Since MRI and CT scans produce 3D-dimensional images. Viewing them can be a challange as viewing 3d images on a 2d plane is not possible without doing something clever. This notebook([medicalGifCreation.ipynb](/notebooks/medicalGifCreation.ipynb)) produces .gif files that makes it possible to look at the whole image.

![Brain tumour](/notebooks/images/brain/MultiBrainTumourMask.gif)

Above is a gif of a MRI scan of a brain with a tumour. The dataset used contains 4 different scans(), and a mask with 3 different values(). The 4 different scans are displayed side by side, whilst the mask first shows a combined mask then the 3 other side by side.

The method that allows this is the pillow libary method save(). Where frames is a list of pillow images.
![saveAsGif](/notebooks/images/readmeImages/saveAsGif.png)

Since the image is in 3dimensions it is possible to look at it from multiple directions. By rotating the image viewd as a matrix we can look at it front-to-back instead of top-to-down.

![Front-To-Back](/notebooks/images/brain/FrontalMaskCombined.gif)

Originaly the brain is facing right, and the image was shown down-to-up. Changing the orientation so the image is facing down can be done by rotaing it. By changing from down-to-up to up-to-down, the image is accidentaly mirrored meaning we have to mirror it back to make sure the tumour is on the correct side. 

Below is the function used to rotate the image so its facing forward.

![RotateFunctio](/notebooks/images/readmeImages/rotateFunction.png)

---
## Segmentation

### Prostate 
[Prostate Segmentation notebooks](/notebooks/prostateSegmentation.ipynb)

Same method from [View Data](#view-data). Image consists of 2 different MR scans and 2 different masks. 

![prostate](/notebooks/images/prostate/ProstateSideBySide.gif)



![prostate0.55Result](/notebooks/images/readmeImages/prostate0.55Result.png)

Image above shows the result when dice score is 0.62 after 20 epochs. As we can see the model predicts outside of where the prostate is. A way to prevent this error could be to add a tranformer that crops the image, as most of the content on the image is not relevant to the segmentation task. 

![prostate00Result](/notebooks/images/readmeImages/prostate1.01Result.png)

The crop size used in this case is [160.0, 160.0, 16.0] as opposed to the orininal [320.0, 320.0, 16.0]. By doing this the image almost only consists of the prostate making the model much more accurate. The result is a dice score of 1.01.

On the image above it looks like its unable to predict the second class well. To look for the imbalance in the mask we can do as done below.

![ClassBalanceTest](/notebooks/images/readmeImages/prostateClassBalanceTest.png)

It is approximately 3 times more of class 1 than class 2. Then we want to see how this compares to a mask prediction.

![maskLabelComparison](/notebooks/images/readmeImages/maskLabelComparison.png)

The first image is predicted mask, and the second image is mask label. The first image contains float values ranging from about -50 to 150. The values should only contain values 0,1,2. The reason i think this is happening is that model parameters are wrong. Not sure how to fix this, others have used "out_channels=number_of_classes" but this gave tensor "AssertionError: ground truth has different shape (torch.Size([4, 1, 160, 160, 32])) from input (torch.Size([4, 3, 160, 160, 32]))". Which i am not able to fix:)

The idea next was to change loss functions and transformations based on how good it was to segment each class. But since it is not multiclass for some reason this is not possible.


---
## Classification

### Intracranial Aneurysm

[Intercranial-Aneruysm-Classification](/notebooks/IntracranialAneurysmClassification.ipynb)

The dataset used is MedNIST intercranial aneurysm, meaning ballooning of blood vessels in the brain. Images are in the shape (28,28,28) and the task is to classify if there is an aneruysm or not(binary classification task).

![image](/notebooks/images/readmeImages/classificationImages.png)

Images above shows how the dataset looks like, it shows blood vessels in the brain.

This dataset is unbalanced, total in training is 1335, and number of images containing aneurysm is 150 or 11% of traning dataset.


Creating a model and learner is shown below...

![Classification-Model-Learner](/notebooks/images/readmeImages/classificationModelLearner.png)

The parameter for the Classifier model "in_shape" requires a list of integers representing the shape of the image. The list should be[channel, depth, height, width], where channel can be seen as the same as different color layers in RGB images. For these images the "in_shape" is [1,28,28,28]

The results is as follows:

![Classification-Results](/notebooks/images/readmeImages/classificationResults.png)

![Classification-Confusion-Matrix](/notebooks/images/readmeImages/classificationConfusionMatrix.png)

The confusion matrix shows clearly that the dataset is unbalanced. The confusion matrix also shows that the model misclassified 10 aneurysm cases and only got 3 aneurysm cases correct. This is a good example that accuracy is not always a good metric to use as it gives almost 87%, but the accuracy for anuerysm is 23%

Since the classes are unbalanced, something that might be worth trying to remove the imbalance is downsampling. Aneurysm class contains 150 cases, so by only keeping 150 cases of non-anuerysm the new dataset has a size of 300.


![downsample](/notebooks/images/readmeImages/downsample.png)

Above is the method that perfomed the downsample. The result is now:

![downsample-result](/notebooks/images/readmeImages/downsampleResult.png)

This is a huge improvement in cases that contains aneurysm. The overall accuracy went down to about 68%, but the precision on aneurysm is now about 67%. 

Otherways to improve an unbalanced dataset could be to use Focal loss, but for some reason i was unable to use it.
