from scipy.stats import zscore
import numpy as np

def remove_outliers(data, threshold = 5):
    ## remove rows that contain outliers
    ## outliers have zscore > threshold
    # take uniques of all the row indices of each px whose zscore is above threshold
    rows_to_delete = list(set([el[0] for el in np.argwhere(abs(zscore(data))>threshold)]))
    return np.delete(data, rows_to_delete, axis=0)

def calc_HHcorr(data):
    N_Pixel = data.shape[1]
    HHcorr = np.zeros(N_Pixel, dtype='float')

    for px_dist in range(0, N_Pixel-1):
        shifted_data = data[:,px_dist+1:].astype(float)
        data_section = data[:,:-px_dist-1].astype(float)
 
        difference = (data_section-shifted_data)**2
        HHcorr[px_dist] = np.mean(difference)
    return HHcorr

        
def calc_autocorr(data):
    N_Pixel = data.shape[1]
    autocorr = np.zeros(N_Pixel, dtype='float')
    
    for px_dist in range(0, N_Pixel-1):
        shifted_data = data[:,px_dist+1:].astype(float)
        data_section = data[:,:-px_dist-1].astype(float)

        product = (data_section*shifted_data)
        autocorr[px_dist]= 2*np.mean(product)
    return autocorr