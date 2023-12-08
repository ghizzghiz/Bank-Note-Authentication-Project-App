# 1. Import the required libraries:
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
    # 2. Load the image in grayscale, resize, and change resolution
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (400, 400))
    normalized_image = (image - np.mean(X_means+image)) / np.mean(X_stds+image) # CHECK AGAIN
    normalized_image = Image.fromarray(normalized_image)
    st.image(image)
    #### keep in mind the size of the images and quality
    # https://auth0.com/blog/image-processing-in-python-with-pillow/
    #image = change_dpi_array(image, 660)
    
    # 3. Perform the wavelet transform on the image:
    coeffs = pywt.wavedec2(normalized_image, 'haar')  # Choose your desired wavelet (e.g., 'haar')
    #coeffs

    # 4. Extract the approximation (low-frequency) and detail (high-frequency) coefficients:
    #approx_coeffs, detail_coeffs = coeffs[0], coeffs[1:]

    # 5. Calculate the desired statistical features for each coefficient level:
    #variance_ = [np.var(level) for level in detail_coeffs]
    #skewness_ = [skew(level).flatten() for level in detail_coeffs]
    #kurtosis_ = [kurtosis(level).flatten() for level in detail_coeffs]
    #entropy_ = [entropy(np.abs(level).flatten()) for level in detail_coeffs]

    # 6. Optionally, you can also calculate these features for the approximation coefficients:
    #approx_variance = np.var(approx_coeffs)
    #approx_skewness = skew(approx_coeffs.flatten())
    #approx_kurtosis = kurtosis(approx_coeffs.flatten())
    #approx_entropy = entropy(np.abs(approx_coeffs.flatten()))

    #result = [approx_variance, approx_skewness, approx_kurtosis, approx_entropy]
    ##result = [variance_, skewness_, kurtosis_, entropy_] # WRONG


    ##st.write("Image Statistics - Mean:", np.mean(image), "Std Dev:", np.std(image))
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs

    # Extract features
    features = {
        'Variance': np.var(cA),
        'Skewness': skew(cA.flatten()),
        'Kurtosis': kurtosis(cA.flatten()),
        'Entropy': entropy(cA.flatten())
    }
    
    return features

# 7. You can then use these calculated features for further analysis or visualization.

# Make sure to adjust the wavelet type and other parameters according to your specific requirements.
# Additionally, you may need to normalize or scale the features depending on your application.
