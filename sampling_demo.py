
#**************************************************************
#**************************************************************
#**************************************************************
# 
# - This Python file contains a demonstration of the Bayesian 
# maximum variance inclusion (BMVI) sampling. The BMVI is 
# compared against simple random sampling (SRS) with synthetic
# data generated from a random Gaussian mixture module function.
# Both samplers are tested with the standard regularized least 
# squares (RLS) prediction method with a linear kernel. 
# Prediction performance and hyperparameter selection is 
# implemented via 10-fold cross-validation. 
# 
# If one wishes to implement BMVI via other prediction methods
# then the respective Hessian and gradient (see the corresponding
# article) needs to be edited. 
#
#**************************************************************
#**************************************************************
#**************************************************************

# STEP 1: Import the relevant modules. 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bmvi_toolbox import *

# STEP 2: Load the synthetic data set generated from a random Gaussian mixture model function.

# Data contains spatial autocorrelation thus mimicking natural phenomena. 
X = np.loadtxt('X.txt', delimiter=',')
y = np.loadtxt('y.txt', delimiter=',')

# For plotting the input data, create meshgrids from the data.  
x_grid = np.reshape(X[:,1], (30,30))
y_grid = np.reshape(X[:,2], (30,30))
z_grid = np.reshape(y, (30,30))
# Plot the synthetic data in a 3D-plot for illustration. 
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid, cmap='jet')
ax.set_xlabel('Predictor feature 1 value')
ax.set_ylabel('Predictor feature 2 value')
ax.set_zlabel('Function value')
ax.set_title('Synthetic data set function (GMM)')
ax.view_init(azim=45, elev=45)
plt.show()

# STEP 3: Implement SRS and BMVI sampling with the synthetic data using RLS prediction model. 

# Do sampling. 
print("Starting SRS sampling...\n")
SRS_inclusion_index_set, SRS_generalization_error_list = SRS(X, y, len(y))
print("Finished SRS sampling.\n")
print("Starting BMVI sampling...\n")
BMVI_inclusion_index_set, BMVI_generalization_error_list = BMVI(X, y, len(y))
print("Finished BMVI sampling, plotting the results...\n")

# Plot the sampling performance results. 
fig = plt.figure(figsize=(10, 7))
plt.plot(range(0, len(SRS_generalization_error_list)), SRS_generalization_error_list, linewidth=2)
plt.plot(range(0, len(BMVI_generalization_error_list)), BMVI_generalization_error_list, 'r', linewidth=2)
plt.grid(True)
ax = plt.gca()
ax.set_xlabel('Number of sampled data points')
ax.set_ylabel('Average absolute prediction error')
ax.set_title('Comparison of SRS/BMVI sampling methods with synthetic data')
ax.legend(('SRS', 'BMVI'))
plt.show()



