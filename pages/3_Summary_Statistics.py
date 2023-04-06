import numpy as np
import pandas as pd
import streamlit as st


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_URL = ('https://raw.githubusercontent.com/claudio1975/Insurance_Cross_Sell_Prediction_Web_App/main/train_small_update.csv')
df = pd.read_csv(DATA_URL)

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df)

# Formatting features
df['Driving_License'] = df['Driving_License'].astype('object')
df['Region_Code'] = df['Region_Code'].astype('object')
df['Previously_Insured'] = df['Previously_Insured'].astype('object')
df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('object')
df['Response'] = df['Response'].astype('object')

st.title("Summary Statistics")


st.subheader('Categorical Variables')
# Summarize attribute distributions for data type of variables
obj_cols = [var for var in df.columns if df[var].dtype=='object']
df[obj_cols].describe().T


st.subheader('Numerical Variables')
# Summarize attribute distributions for data type of variables
num_cols = [var for var in df.columns if df[var].dtype!='object']
df[num_cols].describe().T
