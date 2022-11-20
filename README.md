# ``silver-system``, version 2022.1103 - analysis of receptive fields in deep neural networks
M.Sc. in Computer Science (Bioinformatics and Modeling track - BIM)  
Second semester research project - Sorbonne Université, 2021/2022

Supervisor: Denis Sheynikhovich, Associate Professor - Silver Sight Team, Institut de la vision, Sorbonne Université / INSERM / CNRS  
Authors: Kh&#x00E1;nh Nam NGUY&#x1EC4;N and David Lambert

- _Ab initio_ implementation of an [explainable artificial intelligence](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence) (XAI) method based on discrepancy maps and receptive field computation in convolutional neural networks from [Zhou et al, 2015](http://arxiv.org/abs/1412.6856)

- application to AlexNet and a variant trained by the Silver Sight team on images from a behavioral neuroscience experiment on human orientation in an artificial environment


## Notes
Throughout the code and accompanying documents, mentions of "AlexNet", "AxN", "alexnet", or "axn" refer to the pre-trained, "vanilla" version of AlexNet made available by PyTorch via their Hub platform and downloaded from said platform; "AvatarNet", "AvN", "avatarnet" and "avn" refer to the the network trained by the Avatar team before the authors were onboarded on the present research project.

Each network uses a specific dataset, with specific image sizes, on which a specific preprocessing function must be applied before passing any image through the network.


## Contents
Following the "tracer bullet" principle outlined in [The Pragmatic Programmer](https://pragprog.com/titles/tpp20/the-pragmatic-programmer-20th-anniversary-edition/), this code repository contains a proof-of-concept implementation of the methods outlined in _Zhou et al, 2015_.

The code included is the authors' honest attempt at implementing these methods from scratch, without referring to previously implemented versions of Zhou's or similar methods, and without any prior knowledge of the PyTorch platform.

As a consequence, it is the authors' opinion that, while functional and successful, several aspects of the implementation disqualify it from being production-ready, in particular when it comes to its execution speed. Tentative solutions to the issues identified by the authors have been included in the accompanying documentation (see ``TODO.md``).

The implementation is described in the ``Pipeline`` section below.

The authors would like to stress the fact that the research project, the code repository and the accompanying documents do not deal with dataset curation and/or network training, both tasks having been carried out by members of the Avatar team before the authors joined them for this research project. A brief account of the training process has nevertheless been graciously provided by Bilel Abderrahmane Benziane and included in the report.


## Pipeline
``activation_logging.py`` logs the top 10 images for all units of all ReLU layers of the chosen network

``main_script.py`` wraps the rest of the pipeline: it is where the parameters (which model to use, which units of which layers to target, which stride to use in the occlusion process) for the receptive field computation are defined, and then passed to the scripts listed below:
- ``occlusion.py`` creates the occluded images from the top-10 images, with which the discrepancy map will be computed;
- ``discrepancy.py`` computes the discrepancy maps;
- ``top10.py`` assembles the top-10 images into one to make the results more readable;
- ``rec_field.py`` computes the receptive field from the discrepancy maps.

``helpers.py`` contains the helper functions used in the other scripts to make code more modular. In particular, ``top_10`` parses a given log file and returns the top 10 images for a specific unit of a specific layer; ``stack`` stacks a list of images together, either in a row or in a column; ``occlusion`` creates the occluded images from a specific image; ``network_forward_pass`` passes a given image through a given model.

## Setup and execution (using Ubuntu inside Windows WSL; tests in other environments pending)
The following instructions assume you have created a directory for the project, and are running a CLI inside said directory.

- create a virtual environment to isolate the project's execution and avoid version conflicts, using Python 3.8 or above, for example using ``venv`` (see [documentation](https://docs.python.org/fr/3/library/venv.html)). The following command creates an environment called ``env_silver-system``:
```python3.8 -m venv env_silver-system```
- activate the environment: ```source env_silver-system/bin/activate```
- once the environment has been activated, install the Python dependencies described in the attached ``requirements.txt`` file (for example by calling ``pip install -r requirements.txt``)

You can then run the scripts in the ``./src`` subdirectory:
- ``python activation_logging.py DATASET [NETWORK]`` will go through the files in ``DATASET`` (see ``Datasets`` section below) and log the top 10 activating files for all units of all ReLU layers of ``NETWORK``. The ``NETWORK`` argument is optional: the script will use AlexNet by default, but ``avn`` will specify the use of AvatarNet:  
``python activation_logging.py coco-unlabeled2017``  
``python activation_logging.py avatar_dataset avn``  
- ``python main_script.py`` will execute the rest of the pipeline using either the ``axn`` or the ``avn`` log from the ``logs`` directory, targeting the layers and units specified in its body.

The execution of ``activation_logging.py`` being, at the time of the present release, quite slow, logs for ``axn`` and ``avn`` have already been included in the present repository, under the ``logs`` directory.

AvatarNet should be in an ``avatar`` directory at the same arborescence level as ``src`` and ``logs``: specifically, the file that the scripts expect to find there is named ``alexnet_places_old.pt``.

Directory structure for the datasets is detailed below.


## Datasets
They should be subfolders of the ``./datasets`` folder (paths such as ``./datasets/{DATASET}``). Each of the two networks uses two datasets : a very small one to test and iterate without wasting time while implementing the elementary components and functionalities, and a large one on which the actual results (see accompanying slides and report) are obtained.

A validation subset of the Avatar dataset is used to compute AvatarNet's confusion matrix and accuracy to crosscheck against the figures provided by the team that trained the network and make sure preprocessing worked as planned. Please note that neither AvatarNet nor the Avatar dataset have, to date, been released to the public by the Avatar team.

Below are details about each dataset or set of images:

### COCO dataset (123k images)
Entire unlabeled COCO (Common Objects in COntext) dataset  
[Download link](http://images.cocodataset.org/zips/unlabeled2017.zip)  
Structure: none (flat folder containing the images, no subfolder)

### AlexNet script tests dataset
Subset of the dataset above
Structure: none (flat folder containing the images, no subfolder)

### Avatar dataset
Contents of "DataAfterWellSamplingCleaned.zip"
53k images: 26k in the "geo" class and 27k in the "lmk" class
Structure: two subfolders "geo" and "lmk"

### Avatar script tests dataset
Subset of the above
100 images: 50 in the "geo" class and 50 in the "lmk" class
Structure: two subfolders "geo" and "lmk"

### Avatar validation subset
Contents of the validation subset included in "DataAfterWellSamplingCleanedTrainTestVal.zip", used for the confusion matrix and accuracy computations
1k images: 500 for each class
Structure: two subfolders "geo" and "lmk"
