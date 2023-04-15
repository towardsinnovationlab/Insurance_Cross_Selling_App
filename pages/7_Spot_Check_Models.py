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
from sklearn.calibration import CalibratedClassifierCV
#import joblib
import pickle5 as pickle
import requests
#from io import BytesIO


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

st.title("Spot Check Models")

DATA_URL = ('https://raw.githubusercontent.com/towardsinnovationlab/Insurance_Cross_Selling_App/main/train_small_update.csv')
df = pd.read_csv(DATA_URL)

DATA_URL_xtr = ('https://raw.githubusercontent.com/towardsinnovationlab/Insurance_Cross_Selling_App/main/X_train.csv')
X_train = pd.read_csv(DATA_URL_xtr)
DATA_URL_xte = ('https://raw.githubusercontent.com/towardsinnovationlab/Insurance_Cross_Selling_App/main/X_test.csv')
X_test = pd.read_csv(DATA_URL_xte)
DATA_URL_ytr = ('https://raw.githubusercontent.com/towardsinnovationlab/Insurance_Cross_Selling_App/main/y_train.csv')
y_train = pd.read_csv(DATA_URL_ytr)
DATA_URL_yte = ('https://raw.githubusercontent.com/towardsinnovationlab/Insurance_Cross_Selling_App/main/y_test.csv')
y_test = pd.read_csv(DATA_URL_yte)


# LR Calibration models
# Download the model file from the GitHub repository and read it into a memory buffer:
LR_url = 'https://github.com/towardsinnovationlab/Insurance_Cross_Selling_App/raw/main/LR_C_model.sav'
LR_response = requests.get(LR_url)
LR_model_buf = LR_response.content
# Load the pre-trained model from the memory buffer:
LR_C_restored_model = pickle.load(LR_model_buf)
# Make predictions
predictions_tr = LR_C_restored_model.predict_proba(X_train)[:, 1]
predictions_t = LR_C_restored_model.predict_proba(X_test)[:, 1]
LR_auc_train = roc_auc_score(y_train, predictions_tr)  
LR_auc_test = roc_auc_score(y_test, predictions_t) 
score= {'model_c':['LR_C'], 'auc_train_c':[LR_auc_train],'auc_test_c':[LR_auc_test]}
LR_score= pd.DataFrame(score)

# KNB Calibration models
# Download the model file from the GitHub repository and read it into a memory buffer:
KNB_url = 'https://github.com/towardsinnovationlab/Insurance_Cross_Selling_App/raw/main/KNB_C_model.sav'
KNB_response = requests.get(KNB_url)
KNB_model_buf = KNB_response.content
# Load the pre-trained model from the memory buffer:
KNB_C_restored_model = pickle.load(KNB_model_buf)
# Make predictions
predictions_tr = KNB_C_restored_model.predict_proba(X_train)[:, 1]
predictions_t = KNB_C_restored_model.predict_proba(X_test)[:, 1]
KNB_auc_train = roc_auc_score(y_train, predictions_tr)  
KNB_auc_test = roc_auc_score(y_test, predictions_t) 
score= {'model_c':['KNB_C'], 'auc_train_c':[KNB_auc_train],'auc_test_c':[KNB_auc_test]}
KNB_score= pd.DataFrame(score)

# GNB Calibration models
# Download the model file from the GitHub repository and read it into a memory buffer:
GNB_url = 'https://github.com/towardsinnovationlab/Insurance_Cross_Selling_App/raw/main/GNB_C_model.sav'
GNB_response = requests.get(GNB_url)
GNB_model_buf = GNB_response.content
# Load the pre-trained model from the memory buffer:
GNB_C_restored_model = pickle.load(GNB_model_buf)
# Make predictions
predictions_tr = GNB_C_restored_model.predict_proba(X_train)[:, 1]
predictions_t = GNB_C_restored_model.predict_proba(X_test)[:, 1]
GNB_auc_train = roc_auc_score(y_train, predictions_tr)  
GNB_auc_test = roc_auc_score(y_test, predictions_t) 
score= {'model_c':['GNB_C'], 'auc_train_c':[GNB_auc_train],'auc_test_c':[GNB_auc_test]}
GNB_score= pd.DataFrame(score)

# HGBM Calibration models
# Download the model file from the GitHub repository and read it into a memory buffer:
HGBM_url = 'https://github.com/towardsinnovationlab/Insurance_Cross_Selling_App/raw/main/HGBM_C_model.sav'
HGBM_response = requests.get(HGBM_url)
HGBM_model_buf = HGBM_response.content
# Load the pre-trained model from the memory buffer:
HGBM_C_restored_model = pickle.load(HGBM_model_buf)
# Make predictions
predictions_tr = HGBM_C_restored_model.predict_proba(X_train)[:, 1]
predictions_t = HGBM_C_restored_model.predict_proba(X_test)[:, 1]
HGBM_auc_train = roc_auc_score(y_train, predictions_tr)  
HGBM_auc_test = roc_auc_score(y_test, predictions_t) 
score= {'model_c':['HGBM_C'], 'auc_train_c':[HGBM_auc_train],'auc_test_c':[HGBM_auc_test]}
HGBM_score= pd.DataFrame(score)

score_cal = LR_score.append(KNB_score)
score_cal = score_cal.append(GNB_score)
score_cal = score_cal.append(HGBM_score)
score_cal

# Plot results for a graphical comparison
print("Spot Check Models")
plt.rcParams['figure.figsize']=(15,5)
fig = plt.figure()
plt.subplot(1,2,1)  
sns.stripplot(x="model_c", y="auc_train_c",data=score_cal,size=15)
plt.xticks(rotation=45)
plt.title('Train results')
axes = plt.gca()
axes.set_ylim([0,1.1])
plt.subplot(1,2,2)
sns.stripplot(x="model_c", y="auc_test_c",data=score_cal,size=15)
plt.xticks(rotation=45)
plt.title('Test results')
axes = plt.gca()
axes.set_ylim([0,1.1])
st.pyplot(fig)


st.markdown("""
Logistic Regression (LR) is the benchmark model used in Insurance, and it has been compared with

Naive Bayes model (GNB), K-Nearest Neighbors model (KNB) and Hist Gradient Boosting Machine (HGBM).

""")
