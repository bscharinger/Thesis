# Multicenter Aortic Vessel Tree Extraction Using Deep Learning 

This is the repository associated to the bachelor's thesis "Multicenter Aortic Vessel Tree Extraction Using Deep Learning" of Bernhard Scharinger. Here you can find all Python scripts, notebooks and models used in the thesis.

## What file is used for what?
- [main.ipynb](/main.ipynb): This is the main file of the project.
- [data_gen.py](/data_gen.py): Contains the Keras Sequence-Class, used to feed the network with data.
- [generate_patches.ipynb](/generate_patches.ipynb): This file contains the script to generate our input data patches from the CTA data
- [helper_funcs.py](/helper_funcs.py): Contains different utility functions, like loss-functions, connected component anlysis, and functions for zero-padding.
