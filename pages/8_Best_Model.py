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

# Split data set between target variable and features
X_full = df.copy()
y = X_full.Response
X_full.drop(['Response'], axis=1, inplace=True)

st.title("Best Model Prediction: Hist Gradient Boosting Machine")

# Select numerical columns
X_full.drop(['id'], axis=1, inplace=True)
numerical_cols = [var for var in X_full.columns if X_full[var].dtype in ['float64','int64']]
num = X_full[numerical_cols]

# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [var for var in X_full.columns if
                    X_full[var].nunique() <= 15 and 
                    X_full[var].dtype == "object"]
# Subset with categorical features
cat = X_full[categorical_cols]

# Numerical Features Pre-Processing
num_o = num.copy()
# outliers correction
for col in num_o.columns:
    q75, q25 = np.percentile(num_o[col].dropna(), [75 ,25])
    iqr = q75 - q25 
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5) 
    num_o[col].mask(num_o[col]<min, min, inplace=True)
    num_o[col].mask(num_o[col]>max, max, inplace=True)

# Categorical Features Pre-Processing
# Transform in integer binary variables
y = y.astype('int')
cat['Driving_License'] = cat['Driving_License'].astype('int')
cat['Previously_Insured'] = cat['Previously_Insured'].astype('int')

cat2 = pd.concat([y,cat], axis=1)
cat2['Gender']=cat2['Gender'].map({'Female':0,'Male':1})
cat2['Vehicle_Damage']=cat2['Vehicle_Damage'].map({'No':0,'Yes':1})
# calculate the mean target value per category for each feature and capture the result in a dictionary 
Vehicle_Age_LABELS = cat2.groupby(['Vehicle_Age'])['Response'].mean().to_dict()
# replace for each feature the labels with the mean target values
cat2['Vehicle_Age'] = cat2['Vehicle_Age'].map(Vehicle_Age_LABELS)
# Look at the new subset
target_cat = cat2.drop(['Response'], axis=1)
# Grasp all
X_all = pd.concat([num_o,target_cat], axis=1)

# Find features with variance equal zero 
to_drop = [col for col in X_all.columns if np.var(X_all[col]) == 0]
# Drop features 
X_all_v = X_all.drop(X_all[to_drop], axis=1)

# Correlation 
corr_matrix = X_all_v.corr(method ='spearman')
# Select correlated features and removed it
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Find index of feature columns with correlation greater than 0.80
to_drop = [column for column in upper.columns if any(upper[column].abs() > 0.80)]
# Drop features 
X_all_c = X_all_v.drop(X_all_v[to_drop], axis=1)

# Normalization 
scaling = MinMaxScaler()
# Normalization of numerical features
num_sc = pd.DataFrame(scaling.fit_transform(X_all_c[['Age','Annual_Premium','Vintage']]), columns= ['Age','Annual_Premium','Vintage'])
# Grasp all
X_all_sc = pd.concat([num_sc, X_all_c[['Gender','Driving_License','Previously_Insured']]], axis=1)

# Split data set
# Break off train and test set from data
X_train, X_test, y_train, y_test = train_test_split(X_all_sc, y, train_size=0.8, test_size=0.2,stratify=y,random_state=0)

# Model
HGBM_ = HistGradientBoostingClassifier(random_state=0,learning_rate=0.02, max_bins=80, max_depth= 10)
# Define the model
HGBM_model_ = HGBM_.fit(X_train, y_train)

# prediction
predictions_tr = HGBM_model_.predict(X_train)
predictions_tr_ = pd.DataFrame(predictions_tr, columns=['y_train_pred'])
predictions_te = HGBM_model_.predict(X_test)
predictions_te_ = pd.DataFrame(predictions_te, columns=['y_test_pred'])

# Evaluation
auc_train = roc_auc_score(y_train, HGBM_model_.predict_proba(X_train)[:, 1])  
auc_test = roc_auc_score(y_test, HGBM_model_.predict_proba(X_test)[:, 1]) 

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
plot_roc_auc(y_train, HGBM_model_.predict_proba(X_train)[:, 1], 2)
print('ROC on test')
plot_roc_auc(y_test, HGBM_model_.predict_proba(X_test)[:, 1], 2)
