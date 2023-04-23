import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import scipy.stats as stats
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pickle5 as pickle

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


st.title("Profiling Interested Customers")

df = pd.read_csv('./data/train.csv')

# Formatting features
df['Driving_License'] = df['Driving_License'].astype('object')
df['Region_Code'] = df['Region_Code'].astype('object')
df['Previously_Insured'] = df['Previously_Insured'].astype('object')
df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('object')
df['Response'] = df['Response'].astype('object')
df.drop(['id'], axis=1, inplace=True)

numerical_cols = [var for var in df.columns if df[var].dtype in ['float64','int64']]

df_1 = df.copy()

df_1 = df_1[df_1['Response']==1]

# Select numerical columns
num_1 = df_1[numerical_cols]

# Standardization of data
sc = StandardScaler()
num_sc = sc.fit_transform(num_1)

# loading in the model to predict on the data
with open('./data/kmeans.pkl', 'rb') as pickle_in:
    kmeans = pickle.load(pickle_in)

labels = kmeans.predict(num_sc)
cluster_num = num_1.copy()
cluster_num['kmeans_cluster'] = labels

df_cluster = pd.concat([df_1, cluster_num['kmeans_cluster']], axis=1)

# Group clusters by Annual Premium
df = df_cluster.groupby(df_cluster['kmeans_cluster'], as_index=False)['Annual_Premium'].sum()
df['PERCENTAGE'] = df['Annual_Premium']/df['Annual_Premium'].sum()*100
# dropping not matching rows
df = df.dropna()
# ranking 
df = df.sort_values(by = 'Annual_Premium', ascending = False).reset_index(drop=True)
df_AP = df.style.background_gradient(cmap='winter').format({'PERCENTAGE': "{:.2f}"}).format({'Annual_Premium':"{:,.2f}"})
df_AP



