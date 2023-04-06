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

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_URL = ('https://raw.githubusercontent.com/claudio1975/Insurance_Cross_Sell_Prediction_Web_App/main/train_small_update.csv')
df = pd.read_csv(DATA_URL)

# Formatting features
df['Driving_License'] = df['Driving_License'].astype('object')
df['Region_Code'] = df['Region_Code'].astype('object')
df['Previously_Insured'] = df['Previously_Insured'].astype('object')
df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('object')
df['Response'] = df['Response'].astype('object')


st.title("Profiling Interested Customers")

numerical_cols = [var for var in df.columns if df[var].dtype in ['float64','int64']]

df_1 = df.copy()

df_1 = df_1[df_1['Response']==1]

# Select numerical columns
num_1 = df_1[numerical_cols]

# Standardization of data
sc = StandardScaler()
num_sc = sc.fit_transform(num_1)

kmeans = KMeans(n_clusters=4, random_state=0).fit(num_sc)
labels = kmeans.predict(num_sc)

cluster_num = num_1.copy()
cluster_num['kmeans_cluster'] = labels
len(np.unique(kmeans.labels_))

cluster = cluster_num['kmeans_cluster'].value_counts()
#cluster_df = pd.DataFrame(cluster)
#cluster_df.index.name='clusters'
#cluster_df.rename(columns={'kmeans_cluster':'density_kmeans_cluster'}, inplace=True)
#cluster_df

#tsne_num = TSNE(n_components=2, random_state=0).fit_transform(num_sc)
#tsne_num_df = pd.DataFrame(data = tsne_num, columns = ['x','y'], index=num_1.index)
#dff_km = pd.concat([cluster_num['kmeans_cluster'], tsne_num_df], axis=1)
## Show the diagram
#plt.rcParams['figure.figsize']=(10,10)
#fig = plt.figure()
#sns.scatterplot(x='x',y='y',hue='kmeans_cluster',data=dff_km,edgecolor="black")
#plt.title('t-SNE visualization of kmeans clustering on numerical variables')
#st.pyplot(fig)

df_cluster = pd.concat([df_1, cluster_num['kmeans_cluster']], axis=1)

st.subheader('Group clusters by Annual Premium')
# Group clusters by Annual Premium
df = df_cluster.groupby(df_cluster['kmeans_cluster'], as_index=False)['Annual_Premium'].sum()
df['PERCENTAGE'] = df['Annual_Premium']/df['Annual_Premium'].sum()*100
# dropping not matching rows
df = df.dropna()
# ranking 
df = df.sort_values(by = 'Annual_Premium', ascending = False).reset_index(drop=True)
df_AP = df.style.background_gradient(cmap='winter').format({'PERCENTAGE': "{:.2f}"}).format({'Annual_Premium':"{:,.2f}"})
df_AP

st.subheader('Group clusters by Age')
# Group clusters by Age
df = df_cluster.groupby(df_cluster['kmeans_cluster'], as_index=False)['Age'].mean()
df['PERCENTAGE'] = df['Age']/df['Age'].sum()*100
# dropping not matching rows
df = df.dropna()
# ranking 
df = df.sort_values(by = 'Age', ascending = False).reset_index(drop=True)
df_AGE = df.style.background_gradient(cmap='winter').format({'PERCENTAGE': "{:.2f}"}).format({'Age':"{:,.2f}"})
df_AGE

# Select top cluster per Annual Premium and Age
df_cluster_AP = df_cluster[df_cluster['kmeans_cluster']==2].reset_index(drop=True)
df_cluster_AGE = df_cluster[df_cluster['kmeans_cluster']==1].reset_index(drop=True)

st.subheader('Annual Premium distribution vs Gender, Vehicle_Damage, Vehicle_Age for the Top Cluster')
# Plot Annual Premium vs Gender, Vehicle_Damage, Vehicle_Age
fig = plt.figure()
plt.rcParams['figure.figsize']=(20,7)
plt.subplot(1,3,1)
sns.kdeplot(x=df_cluster_AP['Annual_Premium'],hue=df_cluster_AP['Gender'],palette="crest", multiple='stack')
#plt.title('Annual_Premium vs Gender on Top Cluster')
plt.subplot(1,3,2)
sns.kdeplot(x=df_cluster_AP['Annual_Premium'],hue=df_cluster_AP['Vehicle_Damage'],palette="crest", multiple='stack')
#plt.title('Annual_Premium vs Vehicle_Damage on Top Cluster')
plt.subplot(1,3,3)
sns.kdeplot(x=df_cluster_AP['Annual_Premium'],hue=df_cluster_AP['Vehicle_Age'],palette="crest", multiple='stack')
#plt.title('Annual_Premium vs Vehicle_Age on Top Cluster')
st.pyplot(fig)

st.subheader('Age distribution vs Gender, Vehicle_Damage, Vehicle_Age for the Top Cluster')
# Plot Age vs Gender, Vehicle_Damage, Vehicle_Age
fig=plt.figure()
plt.rcParams['figure.figsize']=(20,7)
plt.subplot(1,3,1)
sns.kdeplot(x=df_cluster_AGE['Age'],hue=df_cluster_AGE['Gender'],palette="crest", multiple='stack')
#plt.title('Age vs Gender on Top Cluster')
plt.subplot(1,3,2)
sns.kdeplot(x=df_cluster_AGE['Age'],hue=df_cluster_AGE['Vehicle_Damage'],palette="crest", multiple='stack')
#plt.title('Age vs Vehicle_Damage on Top Cluster')
plt.subplot(1,3,3)
sns.kdeplot(x=df_cluster_AGE['Age'],hue=df_cluster_AGE['Vehicle_Age'],palette="crest", multiple='stack')
#plt.title('Age vs Vehicle_Age on Top Cluster')
st.pyplot(fig)
