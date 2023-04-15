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
import joblib
import requests
from io import BytesIO


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

st.title("Spot Check Models")

DATA_URL = ('./data/train.csv')
df = pd.read_csv(DATA_URL)

DATA_URL_xtr = ('./data/X_train.csv')
X_train = pd.read_csv(DATA_URL_xtr)
DATA_URL_xte = ('./data/X_test.csv')
X_test = pd.read_csv(DATA_URL_xte)
DATA_URL_ytr = ('./data/y_train.csv')
y_train = pd.read_csv(DATA_URL_ytr)
DATA_URL_yte = ('./data/y_test.csv')
y_test = pd.read_csv(DATA_URL_yte)


# LR Calibration models
# Download the model file from the GitHub repository and read it into a memory buffer:
LR_url = './data/LR_C_model.sav'
LR_response = requests.get(LR_url)
LR_model_buf = BytesIO(LR_response.content)
# Load the pre-trained model from the memory buffer:
LR_C_restored_model = joblib.load(LR_model_buf)
# Make predictions
predictions_tr = LR_C_restored_model.predict_proba(X_train)[:, 1]
predictions_t = LR_C_restored_model.predict_proba(X_test)[:, 1]
LR_auc_train = roc_auc_score(y_train, predictions_tr)  
LR_auc_test = roc_auc_score(y_test, predictions_t) 
score= {'model_c':['LR_C'], 'auc_train_c':[LR_auc_train],'auc_test_c':[LR_auc_test]}
LR_score= pd.DataFrame(score)
LR_score

st.markdown("""
Logistic Regression (LR) is the benchmark model used in Insurance, and it has been compared with

Naive Bayes model (GNB), K-Nearest Neighbors model (KNB) and Hist Gradient Boosting Machine (HGBM).

""")
