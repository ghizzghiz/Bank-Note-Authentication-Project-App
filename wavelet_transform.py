# Import the required libraries
from PIL import Image
import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import entropy
import cv2
import streamlit as st

from Classification_Algorithms import X_means, X_stds

def wavelet_trs(image):
    # Load the image in grayscale, resize, and change resolution
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (400, 400))
    st.image(image)

    # Perform the wavelet transform on the image
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs

    # Extract features
    features = {
        'variance': np.var(cA),
        'skewness': skew(cA.flatten()),
        'kurtosis': kurtosis(cA.flatten()),
        'entropy': entropy(cA.flatten())
    }
    
    return features
