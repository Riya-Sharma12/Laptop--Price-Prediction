import numpy as np
import streamlit as st
import pickle
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


# import the model

pipe = pickle.load(open('pipe.pkl','rb'))
dataset = pickle.load(open('dataset.pkl','rb'))


le = LabelEncoder()
dataset['Company'] = le.fit_transform(dataset['Company'])
dataset['TypeName'] = le.fit_transform(dataset['TypeName'])
dataset['cpu brand'] = le.fit_transform(dataset['cpu brand'])
dataset['Gpu brand'] = le.fit_transform(dataset['Gpu brand'])
dataset['os'] = le.fit_transform(dataset['os'])

st.title("Laptop Price Predictor")

# brand

company = st.selectbox('Brand',dataset['Company'].unique())

# type of laptop
type = st.selectbox('Type',dataset['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)' , [2,4,6,12,24,32,64])

# Weight
weight = st.number_input('Weight of the Laptop')

# touchscreen
touchscreen =st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',dataset['cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',dataset['Gpu brand'].unique())

os = st.selectbox('OS',dataset['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    # st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
    
     # Check for NaN or invalid values
    if np.any(np.isnan(query)) or np.any(query == None):
        st.error("Some input values are invalid. Please check your inputs.")
    else:
        # Predict the price and display
        try:
            predicted_price = int(np.exp(pipe.predict(query)[0]))  # Taking the exponentiation as per your earlier logic
            st.title(f"The predicted price of this configuration is ${predicted_price}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")