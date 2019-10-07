# Project 1: Regression analysis and resampling methods
This project investigates different regression methods by studying how they fit to a generated data set as well as real data. The methods investigated includes the Ordinary Least Squares method, Ridge regression and Lasso regression. Polynomials are fitted to the two-dimensional Franke function, as well as real digital terrain data from the Oslo area. The models are then assessed by employing the bootstrap resampling technique.

In order to run the model, the python script main.py is run in the terminal. The program takes one input argument from the command line; 'franke' or 'terrain' depending on what data set the user would like to predict.

main.py requires the user to download plotting_function.py for creating figures, file_handling.py to create the benchmarks and define_colormap.py for plotting the three-dimensional surface plots with the custom colormaps defined in this script. In order to save the figures, the user needs to have an additional folder "Figures/" in the run directory. Similarly, the user needs a folder "Benchmarks/" in order to create the benchmark file.
