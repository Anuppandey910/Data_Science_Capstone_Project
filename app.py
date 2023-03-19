# Importing Libraries

import pandas as pd
import numpy as np
from sklearn import *
import pickle
import streamlit as st

# Loading the saved model
df1 = pickle.load(open('df1.pkl','rb'))
lasso_model = pickle.load(open('lasso.pkl','rb'))

# Title and Header of the page
st.title('Pre-Owned Car Price Prediction')
st.header('Fill the details to Predict the Car Price')


# Features
# ['car_name', 'model', 'variant', 'year', 'selling_price', 'km_driven',
#        'fuel', 'seller_type', 'transmission', 'owner']

# Creating columns on the page
col1,col2,col3 = st.columns(3)

with col1:
    name = st.selectbox('Car Name',df1['car_name'].unique())

with col2:
    model = st.selectbox('Model',df1['model'].unique())
    
with col3:
    variant = st.selectbox('Variant',df1['variant'].unique())
    
col4,col5 = st.columns(2)

with col4:
    year_old = st.number_input('Year_Old',value=0)
    
with col5:
    km_driven = st.number_input('KM Driven',value=0)
    
col6,col7 = st.columns(2)

with col6:
    fuel = st.selectbox('Fuel',df1['fuel'].unique())
    
with col7:
    seller_type = st.selectbox('Seller Type',df1['seller_type'].unique())
    
    
col8,col9 = st.columns(2)

with col8:
    transmission = st.selectbox('Transmission',df1['transmission'].unique())
    
with col9:
    owner = st.selectbox('Owner',df1['owner'].unique())
    
# Predict the car price    
if st.button('Predict Car Price'):

    input_df = pd.DataFrame({'car_name':[name],'model':[model],'variant':[variant],
                             'km_driven':[km_driven],'fuel':[fuel],'seller_type':[seller_type],
                             'transmission':[transmission],'owner':[owner],'year_old':[year_old]})
    
    result = lasso_model.predict(input_df)
    price = result[0]
    st.header('Rs' +  ' ' + str(round(price)))