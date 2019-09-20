#**************************************************************
#**************************************************************
#**************************************************************
# 
# - This Python file contains the functions required for 
# demonstrating the functionality of Bayesian maximum variance
# inclusion (BMVI) sampling method. Implementations for BMVI, LPM and
# SRS are given. The prediction method used with all samplers
# is the standard regularized least squares (RLS) method with 
# a linear kernel. Prediction performance and hyperparameter 
# selection is implemented via 10-fold cross-validation. 
# 
# If one wishes to implement BMVI via other prediction methods
# then the respective Hessian and gradient (see the corresponding
# article) needs to be edited. 
#
#**************************************************************
#**************************************************************
#**************************************************************

import numpy as np

###############################################################
#
# - DESCRIPTION: This function implements the Bayesian maximum
# variance inclusion (BMVI) sampling for a given data set. 
# 
# - INPUTS: 
 # 'X' X contains the input data with first column assumed to 
# be all ones (constant term). Each row corresponds to one
# observation.
# 'y' corresponds to the vector of output values.
# 'n_samples' integer, the number of data points to be sampled. 
#
# - OUTPUTS: 
# 'inclusion_index_set' a list of data indexes sampled.
# 'generalization_error_list' a list of estimated predicton 
# errors as a function of number of sampled data points. 
#
###############################################################
def BMVI(X, y, n_samples):
    unsampled_data_set_indexes = np.array(range(0, len(y)))
    inclusion_index_set = []
    generalization_error_list = []
    posterior_variance_list = None
    if n_samples >= len(y):
        n_samples = len(y)-1
    for sample_ind in range(0, n_samples):
        if np.mod(sample_ind, 100) == 0:
            print('BMVI sampling ' + str(sample_ind+1) + 'th data point (' + str(n_samples+1) + ' in total)')
        if sample_ind == 0: # First sample, random start
            start_ind = np.random.randint(0, len(y), 2) # We need to have two initial samples because of cross-validation
            inclusion_index_set.append(unsampled_data_set_indexes[start_ind][:])
            inclusion_index_set = inclusion_index_set[0]
            inclusion_index_set = inclusion_index_set.tolist()
            unsampled_data_set_indexes = np.delete(unsampled_data_set_indexes, start_ind)
        else: # 
            # Do BMVI selection
            if len(posterior_variance_list) > 0:
                max_var_ind = np.where(posterior_variance_list == np.max(posterior_variance_list))[0][0]
                inclusion_index_set.append(unsampled_data_set_indexes[max_var_ind])
                unsampled_data_set_indexes = np.delete(unsampled_data_set_indexes, max_var_ind) 
        X_sampled = X[inclusion_index_set, :]
        y_sampled = y[inclusion_index_set]
        X_unsampled = X[unsampled_data_set_indexes, :]
        y_unsampled = y[unsampled_data_set_indexes]
        # Train a RLS prediction model on the currently sampled data 
        w_mp, Hessian = solveRLS(X_sampled, y_sampled)
        y_predict = X_unsampled@w_mp
        posterior_variance_list = []
        for i in range(0, X_unsampled.shape[0]):
            posterior_variance_list.append(X_unsampled[i,:]@np.linalg.pinv(Hessian)@np.transpose(X_unsampled[i,:]))
        generalization_error_list.append(np.mean(np.abs(y_predict-y_unsampled)))
    print("\n")
    return inclusion_index_set, generalization_error_list

###############################################################
#
# - DESCRIPTION: This function implements the local pivotal
# method (LPM) for a given data set. 
# 
# - INPUTS: 
 # 'X' X contains the input data with first column assumed to 
# be all ones (constant term). Each row corresponds to one
# observation.
# 'y' corresponds to the vector of output values.
# 'n_samples' integer, the number of data points to be sampled. 
# 'p' initial inclusion probability. This determines how many data 
# points LPM chooses to incluse. 
#
# - OUTPUTS: 
# 'inclusion_index_set' a list of data indexes sampled.
# 'generalization_error_list' a list of estimated predicton 
# errors as a function of number of sampled data points. 
#
###############################################################
def LPM(X, y, n_samples, p):
    unsampled_data_set_indexes = np.array(range(0, len(y)))
    inclusion_index_set = []
    generalization_error_list = []
    # index_probs contains information about selection order and inclusion probabilities. 
    index_probs = np.array([range(0, len(y)), np.ones((len(y)))*p, np.zeros((len(y)))])
    # Temp is used for updating the above variable.  
    temp = index_probs[:, (np.where(index_probs[1,:] > 0) and np.where(index_probs[1,:] < 1))[0]]
    cnt = 0
    # With LPM we first determine the sampling design and the do prediction estimation. 
    while temp.shape[1] > 1 and np.sum(temp[1,:]) >= 1: # Latter condition is required to ensure the loop does not continue indefinitely. 
        rand_ind = np.random.randint(0, temp.shape[1])
        sample_ind = int(temp[0, rand_ind])
        temp = np.delete(temp, rand_ind, axis=1)
        p1 = index_probs[1, sample_ind]
        sample_X = X[sample_ind, :]
        neighbors_X = X[temp[0,:].astype(int), :]
        dist_mat = np.sqrt(np.sum(np.power(neighbors_X-sample_X, 2), axis=1))
        min_dist_index = np.where(dist_mat == np.min(dist_mat))[0][0]
        neighbor_ind = int(temp[0, min_dist_index])
        p2 = temp[1, min_dist_index]
        randNum = np.random.uniform()
        if p1+p2 < 1:
            if randNum < p1/float((p1+p2)):
                p1 = p1+p2
                p2 = 0
            else:
                p1 = 0
                p2 = p1+p2
        elif p1+p2 >= 1:
            if randNum < (1-p1)/float((2-p1-p2)):
                p1 = p1+p2-1
                p2 = 1
            else:
                p1 = 1
                p2 = p1+p2-1
        index_probs[1, sample_ind] = p1
        index_probs[1, neighbor_ind] = p2
        if p1 == 1:
            index_probs[2, sample_ind] = cnt
            cnt += 1
        elif p2 == 1:
            index_probs[2, neighbor_ind] = cnt
            cnt += 1
        temp = index_probs[:, (np.where(index_probs[1,:] > 0) and (np.where(index_probs[1,:] < 1)))[0]]
    # Sample the last indexes last
    unsampled_inds = np.where(index_probs[1,:] < 1)[0]
    cnt = np.max(index_probs[2,:]) + 1
    if unsampled_inds.size > 0:
        cnt = np.max(index_probs[2,:]) + 1
        for ind in unsampled_inds:
            index_probs[1,ind] = 1
            index_probs[2,ind] = cnt
            cnt += 1
    # Sort the data indexes based on selection order. 
    index_probs = index_probs[:, np.argsort(index_probs[2,:])]
    inclusion_index_set = index_probs[0,:]
    # Implement cross-validation and prediction. 
    for i in range(2, len(inclusion_index_set)-1): # We can't sample all data and do prediction estimation. Minumum 2 data points required. 
        if np.mod(i, 100) == 0:
            print('LPM sampling ' + str(i+1) + 'th data point (' + str(len(inclusion_index_set)) + ' in total)')
        X_sampled = X[inclusion_index_set[0:i].astype(int), :]
        y_sampled = y[inclusion_index_set[0:i].astype(int)]
        X_unsampled = X[inclusion_index_set[i:len(inclusion_index_set)].astype(int), :]
        y_unsampled = y[inclusion_index_set[i:len(inclusion_index_set)].astype(int)]
        # Train a RLS prediction model on the currently sampled data 
        w_mp, Hessian = solveRLS(X_sampled, y_sampled)
        y_predict = X_unsampled@w_mp
        generalization_error_list.append(np.mean(np.abs(y_predict-y_unsampled)))
    print("\n")
    return inclusion_index_set, generalization_error_list

###############################################################
#
# - DESCRIPTION: This function implements the simple random 
# sampling (SRS) method for a given data set. 
# 
# - INPUTS: 
 # 'X' X contains the input data with first column assumed to 
# be all ones (constant term). Each row corresponds to one
# observation.
# 'y' corresponds to the vector of output values.
# 'n_samples' integer, the number of data points to be sampled. 
#
# - OUTPUTS: 
# 'inclusion_index_set' a list of data indexes sampled.
# 'generalization_error_list' a list of estimated predicton 
# errors as a function of number of sampled data points. 
#
###############################################################
def SRS(X, y, n_samples):
    unsampled_data_set_indexes = np.array(range(0, len(y)))
    inclusion_index_set = []
    generalization_error_list = []
    if n_samples >= len(y):
        n_samples = len(y)-1
    for sample_ind in range(0, n_samples):
        if np.mod(sample_ind, 100) == 0:
            print('SRS sampling ' + str(sample_ind+1) + 'th data point (' + str(n_samples+1) + ' in total)')
        if sample_ind == 0: # First sample, random start
            start_ind = np.random.randint(0, len(y), 2) # We need to have two initial samples because of cross-validation
            inclusion_index_set.append(unsampled_data_set_indexes[start_ind][:])
            inclusion_index_set = inclusion_index_set[0]
            inclusion_index_set = inclusion_index_set.tolist()
            unsampled_data_set_indexes = np.delete(unsampled_data_set_indexes, start_ind)
        else:
            rand_ind = np.random.randint(0, len(unsampled_data_set_indexes))
            inclusion_index_set.append(unsampled_data_set_indexes[rand_ind])
            unsampled_data_set_indexes = np.delete(unsampled_data_set_indexes, rand_ind)
        X_sampled = X[inclusion_index_set, :]
        y_sampled = y[inclusion_index_set]
        X_unsampled = X[unsampled_data_set_indexes, :]
        y_unsampled = y[unsampled_data_set_indexes]
        # Train a RLS prediction model on the currently sampled data 
        w_mp, Hessian = solveRLS(X_sampled, y_sampled)
        y_predict = X_unsampled@w_mp
        generalization_error_list.append(np.mean(np.abs(y_predict-y_unsampled)))
    print("\n")
    return inclusion_index_set, generalization_error_list

###############################################################
#
# - DESCRIPTION: This function produces a cross-validation 
# fold partitioning. 
# 
# - INPUTS: 
# 'n_samples' integer, the number of data points. 
# 'n_folds' integer, the number of folds.
#
# - OUTPUTS: 
# 'folds' a list of integer lists containing fold indices. 
#
###############################################################
def makeFolds(n_samples, n_folds):
    folds = []
    index_list = np.random.permutation(n_samples)
    # Check that the number of data points is larger than required number of folds
    if n_samples > n_folds:
        fold_size = np.floor(n_samples/float(n_folds))
        for fold in range(0, n_folds):
            start_ind = int(fold_size*fold)
            end_ind = int(fold_size*(fold+1))
            if fold < n_folds-1:
                folds.append(index_list[start_ind:end_ind].tolist())
            else:
                folds.append(index_list[start_ind:].tolist())
        return folds
    # Otherwise, we create a leave-one-out fold partitioning. 
    else:
        for i in range(0, len(index_list)):
            folds.append([index_list[i]])
        return folds
    
###############################################################
#
# - DESCRIPTION: This function solves the maximum likelihood (ML)
# regularized least squares model. Hyperparameter selection 
# is conducted using 10-fold cross-validation. 
# 
# - INPUTS: 
 # 'X' X contains the input data with first column assumed to 
# be all ones (constant term). Each row corresponds to one
# observation.
# 'y' corresponds to the vector of output values.
#
# - OUTPUTS: 
# 'optimal_w_mp' the hyperparameter tuned ML weight vector for 
# RLS model.
# 'Hessian' the optimal Hessian matrix corresponding to 
# matrix A of function S(w) in equation (8) in the article. 
#
###############################################################
def solveRLS(X, y):
    # Choose alpha/beta hyperparameters from an exponential
    # grid.
    alpha_grid = float(2)**np.arange(-7,7)
    beta_grid = float(2)**np.arange(-7,7)
    # Optimal hyperparameters and auxiliary variables
    optimal_alpha = None
    optimal_beta = None
    optimal_error = np.inf
    alphabeta_matrix = np.zeros((X.shape[1], X.shape[1]))
    alphabeta_matrix[1:,1:] = np.eye(X.shape[1]-1)
    # Loop through the hyperparameter grid 
    folds = makeFolds(X.shape[0], 10)
    for alpha in alpha_grid:
        for beta in beta_grid: 
            prediction_error_list = []
            # Choose best hyperparameters via cross-validation
            for fold_ind in range(0, len(folds)):
                ind = folds[fold_ind]
                X_train = X
                y_train = y
                X_train = np.delete(X_train, ind, 0)
                y_train = np.delete(y_train, ind, 0)
                X_test = X[ind,:]
                y_test = y[ind]
                X_train_T = np.transpose(X_train)
                # Solve the maximum likelihood model
                w_mp = np.linalg.inv(X_train_T@X_train + alpha/float(beta)*alphabeta_matrix)@X_train_T@y_train
                # Make prediction to validation data
                y_pred = X_test@w_mp
                prediction_error_list.append(np.sum(np.abs(y_pred-y_test)))
            # Evaluate the prediction error on evaluation data and save the best found parameters
            if np.sum(prediction_error_list) < optimal_error:
                optimal_error = np.sum(prediction_error_list)
                optimal_alpha = alpha
                optimal_beta = beta
    # Get the optimal parameters and return to caller
    X_T = np.transpose(X)
    optimal_w_mp = np.linalg.inv(X_T@X + optimal_alpha/float(optimal_beta)*alphabeta_matrix)@X_T@y
    Hessian = optimal_beta*X_T@X + optimal_alpha*alphabeta_matrix
    return optimal_w_mp, Hessian

