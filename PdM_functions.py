# https://github.com/dganguli/robust-pca
from __future__ import division, print_function

import numpy as np
import pandas as pd
from itertools import groupby

try:
    from pylab import plt
except ImportError:
    print('Unable to import pylab. R_pca.plot_fit() will not work.')

try:
    # Python 2: 'xrange' is the iterative version
    range = xrange
except NameError:
    # Python 3: 'range' is iterative - no need for 'xrange'
    pass


class R_pca:

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        #this loop implements the principal component pursuit (PCP) algorithm
        #located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)                            #this line implements step 3
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)             #this line implements step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)                                      #this line implements step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                print('iteration: {0}, error: {1}'.format(iter, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')

# calculate the covariance matrix
def cov_matrix(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            return covariance_matrix, inv_covariance_matrix
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")
        
# Calculate the Mahalanobis (M) distance
# M = Multi-variate distance measurement
# Underpins the hotelling T-square test
def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    return md

# CHECK IF MATRIX IS POSITIVE DEFINITE
def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


# CALCULATE THRESHOLD FOR CLASSIFYING AS ANOMALY
def MD_threshold(dist, extreme = False, verbose = False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    return threshold

def ifelse(x, list_of_checks, yes_val, no_val):
    if x in list_of_checks:
        res = yes_val
    else: 
        res = no_val
    return(res)

# Clamp data
def clampit(x, cl_lim, cl_val, cl_type):
    if cl_type == 'high':
        if x > cl_lim:
            res = cl_val
        else: 
            res = x
        return(res)
    elif cl_type == 'low':
        if x < cl_lim:
            res = cl_val
        else: 
            res = x
        return(res)

# Check for NAs (missing data)
def check_for_nas(dat):
    for col in dat.columns:
        count_nas = dat[col].isnull().sum()
        count_per = str(round(100 * count_nas / len(dat), 2)) + '%'
        print(str(col) + ' has ' + str(count_nas) + ' NAs\t\t\t' + count_per)  

# Get the cumsum of a list
def Cumulative(lists):
    cu_list = []
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)]
    return cu_list[1:] 


# Define a function to remove segments of timeseries data
# e.g., a large period of broken/recovering time
def remove_segements(segment_vec, mapping, dat, trans_window, keep):    
    print('Input segement categories: ', segment_vec.unique())
    
    # Dataframe to hold the segments that will cut out
    dat_cut = pd.DataFrame()
    
    # Apply to the mapping to the segment vector
    segment_vec = segment_vec.map(mapping)
    print('Categories after mapping: ', segment_vec.unique())
    
    # Calculate the run length encoding
    rle = [(k, sum(1 for i in g)) for k, g in groupby(segment_vec)]
    print('Run length encoding: ', rle)

    # Convert rle to a dataframe
    value = []
    length = []
    for element in range(len(rle)):
        value.append(rle[element][0])
        length.append(rle[element][1])        
        
    cum_length = Cumulative(length)
    df_rle = pd.DataFrame()
    df_rle['value'] = pd.Series(value)
    df_rle['length'] = pd.Series(length)
    df_rle['run_end'] = pd.Series(cum_length)
    df_rle['run_start'] = df_rle['run_end'].shift(1, axis = 0)
    df_rle.iloc[0, -1] = 0
    df_rle['count'] = (df_rle['run_end']+trans_window) - (df_rle['run_start']-trans_window)
   
    # Now loop through and remove rows
    dat.reset_index(drop = True, inplace = True)
    dat['id'] = dat.index # use an id column for row identification
    for row in range(len(df_rle)):     
        print('>')
        if df_rle.loc[row, 'value'] == keep:
            print('Skipping ' + keep + ' segment(s)...')
            next
        else:    
            start_row = int(df_rle.loc[row, 'run_start']) - trans_window
            end_row = int(df_rle.loc[row, 'run_end']) + trans_window - 1
            print('Remove rows: ', [start_row, end_row])
            print('Removing ' + str(end_row - start_row + 1) + ' rows...')  
            
            # Add rows to cut-out dataset
            dat_cut_add = dat[dat['id'].isin(list(range(start_row, (end_row+1), 1)))]
            dat_cut = pd.concat([dat_cut, dat_cut_add], ignore_index=True)

            # Remove rows from base dataset
            dat = dat[dat['id'].isin(list(range(start_row, (end_row+1), 1))) == False]            
    
    # Revert to timestamp index & drop id column
    dat.index = dat['timestamp']
    dat = dat.drop('id', axis = 1)
    dat_cut.index = dat_cut['timestamp']
    dat_cut = dat_cut.drop('id', axis = 1)

    return(dat, dat_cut)

def prep_mahalonobis_data (dist_dat, thresh, pca_dat):
    # Prepare test data for visualization
    dat = pd.DataFrame()
    dat['Mob dist'] = dist_dat
    dat['Thresh'] = thresh
    dat['Anomaly'] = dat['Mob dist'] > dat['Thresh']
    dat.index = pca_dat.index
    dat['timestamp'] = dat.index

    n_outliers = dat[dat['Anomaly'] == True].shape[0]
    print("There are", n_outliers, "anomalies in the test set out of", dat.shape[0], "points")
    print("> Corresponding to " + str(round(100*(n_outliers / dat.shape[0]), 2)) + '%')

    return(dat)














 