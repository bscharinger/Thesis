# Multicenter Aortic Vessel Tree Extraction Using Deep Learning 

This is the repository associated to the bachelor's thesis "Multicenter Aortic Vessel Tree Extraction Using Deep Learning" of Bernhard Scharinger. Here you can find all Python scripts, notebooks and models used in the thesis.

This work is going to be presented in the conference on Biomedical Applications in Molecular, Structural, and Functional Imaging, part of SPIE Medical Imaging: http://spie.org/MI106. You can find the submitted paper [here](https://spie.org/medical-imaging/presentation/Multicenter-aortic-vessel-tree-extraction-using-deep-learning/12468-51).

## Purpose

Pathologies of the cardiovascular system, like dissections and aneurysms, can be life-threatening and require
prompt attention. Therefore automatic segmentation can be a helpful tool to promptly identify an abnormal
anatomy. This process usually requires a significant amount of manual labour with traditional segmentation
methods. To simplify this process, we developed a 3D deep neural network that consists of an encoder-decoder
network together with a self-attention block and evaluated the role of the attention block. A collection of 56
computational tomography angiography (CTA) scans, 4 preprocessed with windowing, re-sampling, cropping and
normalization was used to train, validate and test the networks.

## Data

The data used for this project was a collection of 56 CTA scans obtain from Lukas Radl et al.. The collection can be found [here](https://figshare.com/articles/dataset/Aortic_Vessel_Tree_AVT_CTA_Datasets_and_Segmentations/14806362)

## What file is used for what?
- [main.ipynb](/main.ipynb): This is the main file of the project.
- [data_gen.py](/data_gen.py): Contains the Keras Sequence-Class, used to feed the network with data.
- [generate_patches.ipynb](/generate_patches.ipynb): This file contains the script to generate our input data patches from the CTA data
- [helper_funcs.py](/helper_funcs.py): Contains different utility functions, like loss-functions, connected component anlysis, and functions for zero-padding.
- [Networks](/Networks/): Contains the tested Neural Networks
