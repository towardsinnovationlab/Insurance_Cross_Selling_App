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
import joblib
import pickle5 as pickle

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

st.title("Best Model Prediction: Hist Gradient Boosting Machine")


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


# Model
# Calibration model
# loading in the model to predict on the data
with open('./data/HGBM_tclassifier_.pkl', 'rb') as pickle_in:
    HGBM_tclassifier = pickle.load(pickle_in)

# prediction
predictions_tr = HGBM_tclassifier.predict(X_train)
predictions_tr_ = pd.DataFrame(predictions_tr, columns=['y_train_pred'])
predictions_te = HGBM_tclassifier.predict(X_test)
predictions_te_ = pd.DataFrame(predictions_te, columns=['y_test_pred'])

# Evaluation
auc_train = roc_auc_score(y_train, HGBM_tclassifier.predict_proba(X_train)[:, 1])  
auc_test = roc_auc_score(y_test, HGBM_tclassifier.predict_proba(X_test)[:, 1]) 

# metrics table
d1 = {'evaluation': ['AUC'],
     'model': ['HGBM'],
    'train': [auc_train],
    'test': [auc_test]
        }
df1 = pd.DataFrame(data=d1, columns=['model','evaluation','train','test'])
print('HGBM evaluation on cross-sell prediction')
df1

def plot_roc_auc(y_test, y_score, classes):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.rcParams['figure.figsize']=(10,5)
    fig = plt.figure()
    plt.plot(fpr, tpr,color='orange', lw=2, label='ROC curve (area = {0:0.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1],color='brown', lw=2, linestyle='--' )
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot(fig)

print('ROC on train')
plot_roc_auc(y_train, HGBM_tclassifier.predict_proba(X_train)[:, 1], 2)
print('ROC on test')
plot_roc_auc(y_test, HGBM_tclassifier.predict_proba(X_test)[:, 1], 2)