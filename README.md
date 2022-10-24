# silver system, version 2022.1024 - Analysis of receptive fields in deep neural networks
M.Sc. in Computer Science (Bioinformatics and Modeling track - BIM)  
Second Semester research project - Sorbonne Université, 2021/2022

Supervisor: Denis Sheynikhovich, Associate Professor - Silver Sight Team, Institut de la vision, Sorbonne Université / INSERM / CNRS  
Authors: Khanh Nam Nguyen and David Lambert

- _Ab initio_ implementation of an ![explainable artificial intelligence](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence) (XAI) method based on discrepancy maps and receptive field computation in convolutional neural networks from ![Zhou et al., 2015](http://arxiv.org/abs/1412.6856)

- application to AlexNet and a variant trained by the Silver Sight team on images from a behavioral neuroscience experiment on human orientation in an artificial environment


## Notes
Throughout the report and code, mentions of "AlexNet", "AxN", "alexnet", or "axn" refer to the pre-trained, "vanilla" version of AlexNet made available by PyTorch via their Hub platform and downloaded from said platform; "AvatarNet", "AvN", "avatarnet" and "avn" refer to the the network trained by the Avatar team before the authors were onboarded on the present research project.

Each network uses a specific dataset, with specific image sizes, on which a specific preprocessing function must be applied before passing any image through the network.


## Contents
Following the "tracer bullet" principle outlined in _The Pragmatic Programmer_, this code repository contains a proof-of-concept implementation of the methods outlined in _Zhou et al., 2015_.

The code included is the authors' honest attempt at implementing these methods from scratch, without referring to previously implemented versions of Zhou's or similar methods, and without any prior knowledge of the PyTorch platform.

As a consequence, it is the authors' opinion that, while functional and successful, several aspects of the implementation disqualify it from being production-ready, in particular when it comes to its execution speed. Tentative solutions to the problems identified by the authors have been included in the to-do list and the post-mortem.

The implementation consists in:
- Python scripts that focus on individual steps of the method;
- bash/CLI scripts that chain the Python scripts together to automate the pipeline, as described above.

The authors would like to stress the fact that the research project, the code repository and the accompanying report do not deal with dataset curation and/or network training, both tasks having been carried out by members of the Avatar team before the authors joined them for this research project. A brief account of the training process has nevertheless been graciously provided by Bilel Abderrahmane Benziane and included in the report.


## Pipeline
Python scripts:
``resize.py`` resizes a given image to make it compatible with AlexNet
``activation.py`` logs the top 10 images for all units of all ReLU layers (layers of interest)
``process.py`` applies the ``occlusion`` function to the top-10 images to create the occluded images with which the discrepancy map will be computed.
``top10.py`` assembles the top-10 images into one to make the results more readable.
``helpers.py`` contains only helper functions used in the other scripts to make code more modular.

batch scripts:
``top10.bat`` identifies the top-10 images for the target units and logs the results
``discrepancy.bat`` computes the discrepancy maps based on these logs
The loops in these scripts have the following arguments: ``(start_cell, step, end_cell)``


## Datasets
They should be subfolders of the "./datasets" folder. Each of the two networks uses two datasets : a very small one to test and iterate without wasting time while implementing the elementary components and functionalities, and a large one on which the actual results (see accompanying slides and report) are obtained.

A validation subset of the Avatar dataset is used to compute AvatarNet's confusion matrix and accuracy to crosscheck against the figures provided by the team that trained the network and make sure preprocessing worked as planned.

### COCO dataset (123k images)
Entire unlabeled COCO (Common Objects in COntext) dataset
path: "./datasets/coco-unlabeled2017"

### AlexNet script tests dataset
Subset of the Places365 validation set
path: "./datasets/test_dataset"

### Avatar dataset
Contents of "DataAfterWellSamplingCleaned.zip"
53k images, 26k in the "geo" class and 27k in the "lmk" class
path: "./datasets/avatar_dataset"
structure: two subfolders "geo" and "lmk"

### Avatar script tests dataset
Subset of the above
100 images, 50 in the "geo" class and 50 in the "lmk" class
path: "./datasets/avatar_test"
structure: two subfolders "geo" and "lmk"

### Avatar validation subset
Contents of the validation subset included in "DataAfterWellSamplingCleanedTrainTestVal.zip", used for the confusion matrix and accuracy computations
1k images, 500 for each class
path: "./datasets/avt_test"
structure: two subfolders "geo" and "lmk"
