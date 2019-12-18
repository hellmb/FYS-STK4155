# Project 3: k-means clustering of Ca II 8542 as a solar flare diagnostics tool
## Overview:
In this project, a k-means clustering model is developed and applied to observational data from the Swedish 1-m Solar Telescope (SST). The aim is to cluster pixels of each frame in the time series based on their Ca II 8542 Å line profile in order to detect B-class flares.
## Code:
### Serial:
The serial k-means clustering method is implemented in kmeans_serial.py and run through main_serial.py. 
### Parallel:
The parallel k-means clustering method is implemented in kmeans_parallel.py and run through main_parallel.py. The program takes an input argument specifying the data to run ('simple_data' or 'training_data', but the training data is not present in the folder of this GitHib repository due to reasons explained below). 
## Notes:
The observational data used in this project is not uploaded to GitHub as it is the property of the University of Oslo. The developed models can be run with a simple data set as specified above, or the user can import their own data set. 

In order to save the figures, the user needs to have an additional folder "Figures/" in the run directory. 