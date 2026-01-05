"""
    Helper function for data procssing
"""

import numpy as np 
from scipy.signal import savgol_filter
import pandas as pd


def SNV(data):
    return (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)


# def filter_spectra(spectra_array):
#     """
#     Apply Savitzky-Golay filter to smooth the raman spectra

#     Args:
#         specra_array (2D numpy array): array of input raman spectra
#         window_length (int, optional): length of the filter window. Defaults to 11.
#         polyorder (int, optional): order of the polynomial used to fit the samples. Defaults to 3.

#     Returns:
#         2D numpy array: array of smoothed raman spectra
#     """
    
#     return savgol_filter(specra_array, 50,5)

def segment_spectrum(spectrum, w=50, dw=25):
    """
    Segment the raman spectrum into overlapping windows

    Args:
        spectrum (numpy array): input raman spectrum
        w (int, optional): length of window. Defaults to 50.
        dw (int, optional): step size. Defaults to 25.

    Returns:
        numpy array: array of segmented raman spectrum
    """

    return np.array([spectrum[i:i+w] for i in range(0,len(spectrum)-w,dw) ])

    ### inefficient ###
    #segments = []
    #for i in range(0,len(sig)-w,dw):
    #    segments.append(sig[i:i+w])

    #return np.array(segments)

def segment_spectrum_batch(spectra_mat, w=50, dw=25):
    """
    Segment multiple raman spectra into overlapping windows

    Args:
        spectra_mat (2D numpy array): array of input raman spectrum
        w (int, optional): length of window. Defaults to 50.
        dw (int, optional): step size. Defaults to 25.

    Returns:
        list of numpy array: list containing arrays of segmented raman spectrum
    """
    
    return [spectra_mat[:,i:i+w] for i in range(0,spectra_mat.shape[1]-w,dw) ]

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    train_in = pd.read_csv("Training_inputs.csv")
    # plt.plot("Wave Number", "A_100 c", data = train_in)
    # plt.show()
    
    #train_in = train_in.drop(train_in.columns[0], axis=1)
    #X= train_in.T



    #ti_filtered = train_in[1:].apply(filter_spectra)
    # plt.plot("Wave Number", "A_100 c", data = ti_filtered)
    # plt.show()

    fig, axes = plt.subplots(2,1)
    axes[0].plot("Wave Number", "A_100 c", data = train_in)
    #axes[1].plot("Wave Number", "A_100 c", data = ti_filtered)
    axes[1].set_title("Filtered Spectrum")
    plt.show()