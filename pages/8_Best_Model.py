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
import pickle5 as pickle

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

st.title("Best Model Prediction: Hist Gradient Boosting Machine")

st.markdown("""
Hist Gradient Boosting Machine shows the best performance, then it's been fine tuned and here the results: 
""")


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


# Model
# loading in the model to predict on the data
with open('./data/HGBM_tclassifier.pkl', 'rb') as pickle_in:
    HGBM_tclassifier = pickle.load(pickle_in)
    
# prediction
predictions_tr = HGBM_tclassifier.predict_proba(X_train)[:,1]
predictions_tr_ = pd.DataFrame(predictions_tr, columns=['prediction'])
predictions_te = HGBM_tclassifier.predict_proba(X_test)[:,1]
predictions_te_ = pd.DataFrame(predictions_te, columns=['prediction'])

predictions_tr_ = pd.DataFrame(predictions_tr, columns=['prediction']).reset_index(drop=True)
predictions_te_ = pd.DataFrame(predictions_te, columns=['prediction']).reset_index(drop=True)
predictions = pd.concat([predictions_tr_, predictions_te_], axis=0).reset_index(drop=True)
df = pd.concat([df, predictions], axis=1)
df
#fig=plt.figure()
#sns.countplot(response,df)
#st.pyplot(fig)


# Evaluation
auc_train = roc_auc_score(y_train, predictions_tr)  
auc_test = roc_auc_score(y_test, predictions_te) 

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

st.write('ROC on train')
plot_roc_auc(y_train, predictions_tr, 2)
st.write('ROC on test')
plot_roc_auc(y_test, predictions_te, 2)
