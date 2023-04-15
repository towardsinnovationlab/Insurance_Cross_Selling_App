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
import pickle5 as pickle


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

# LR model
# loading in the model to predict on the data
with open('./data/LR_classifier.pkl', 'rb') as pickle_in:
    classifier = pickle.load(pickle_in)
    
predictions_tr = LR_classifier.predict_proba(X_train)[:, 1]
predictions_t = LR_classifier.predict_proba(X_test)[:, 1]
LR_auc_train = roc_auc_score(y_train, predictions_tr)  
LR_auc_test = roc_auc_score(y_test, predictions_t) 
score= {'model':['LR'], 'auc_train_c':[LR_auc_train],'auc_test_c':[LR_auc_test]}
LR_score= pd.DataFrame(score)

# GNB model
# loading in the model to predict on the data
with open('./data/GNB_classifier.pkl', 'rb') as pickle_in:
    classifier = pickle.load(pickle_in)
    
predictions_tr = GNB_classifier.predict_proba(X_train)[:, 1]
predictions_t = GNB_classifier.predict_proba(X_test)[:, 1]
GNB_auc_train = roc_auc_score(y_train, predictions_tr)  
GNB_auc_test = roc_auc_score(y_test, predictions_t) 
score= {'model':['GNB'], 'auc_train_c':[GNB_auc_train],'auc_test_c':[GNB_auc_test]}
GNB_score= pd.DataFrame(score)

# HGBM model
# loading in the model to predict on the data
with open('./data/HGBM_classifier.pkl', 'rb') as pickle_in:
    classifier = pickle.load(pickle_in)
    
predictions_tr = HGBM_classifier.predict_proba(X_train)[:, 1]
predictions_t = HGBM_classifier.predict_proba(X_test)[:, 1]
HGBM_auc_train = roc_auc_score(y_train, predictions_tr)  
HGBM_auc_test = roc_auc_score(y_test, predictions_t) 
score= {'model':['HGBM'], 'auc_train_c':[HGBM_auc_train],'auc_test_c':[HGBM_auc_test]}
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

Naive Bayes model (GNB), and Hist Gradient Boosting Machine (HGBM).

""")
