# Bayesian maximum variance inclusion

Python and Jupyter notebook implementations of the BMVI sampling method with ridge regression (regularized least squares, RLS) prediction method. The BMVI method is compared against SRS and LPM with a synthetically generated data via Gaussian mixture model function. Prediction performance estimation and hyperparameter selection is implemented via 10-fold cross-validation. 

** Description of the files **

- X.txt -- contains a 900x3 data matrix corresponding to the input predictor features. 
- y.txt -- contains a 900x1 data vector corresponding to the output values from a Gaussian mixture model function. 
- bmvi_toolbox.py/bmvi_toolbox.ipynb -- contains the relevant Python functions needed to implement BMVI sampling and modeling. 
- sampling_demo.py/sampling_demo.ipynb -- contains a Python demo implementation case of the BMVI sampling. 
