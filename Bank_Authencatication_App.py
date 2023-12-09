# streamlit run 0_Bank_Authencaticate_App.py
#import requirements
import streamlit as st
import numpy as np
import pandas as pd
from wavelet_transform import wavelet_trs
from PIL import Image
#import Classification_Algorithms # our Logistic Regression model
#from Classification_Algorithms import LogiReg
from sklearn.preprocessing import StandardScaler

#from sklearn.externals 
import joblib
# Load the saved Logistic Regression model
loaded_model = joblib.load('finalized_LogiReg.sav')

st.title('Banknote Authentication')
st.write("""
         
Upload a picture of a US Banknote and find out if it is authentic or fake. 

""")

uploaded_file = st.file_uploader("Upload an image", accept_multiple_files=False, type = 'jpeg')

if uploaded_file is not None:
    # Before
    st.subheader('Image Before Processing')
    st.image(uploaded_file)
    
    uploaded_file = Image.open(uploaded_file)
    uploaded_file = np.array(uploaded_file)

    # After
    st.subheader('Image After Processing')
    result = wavelet_trs(uploaded_file)
    st.write(result)

    #### SOMETHING WRONG WITH THIS STANDARDIZATION
    #scaler = StandardScaler()
    #scaler.fit(pd.DataFrame(result).transpose())
    #result = scaler.fit_transform(pd.DataFrame(result).transpose())

    #result = pd.DataFrame(result)#.transpose()

    #st.write(result)

    #st.write(result.type)
    #st.write(result.dtypes) # all floats 

    predictions = loaded_model.predict(result)
    st.write('The prediction is that: ')
    st.write(predictions)
