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
import pickle


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

#DATA_URL = ('https://raw.githubusercontent.com/claudio1975/Insurance_Cross_Sell_Prediction_Web_App/main/train_small_update.csv')
#df = pd.read_csv(DATA_URL)

# Formatting features
#df['Driving_License'] = df['Driving_License'].astype('object')
#df['Region_Code'] = df['Region_Code'].astype('object')
#df['Previously_Insured'] = df['Previously_Insured'].astype('object')
#df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('object')
#df['Response'] = df['Response'].astype('object')

# Split data set between target variable and features
#X_full = df.copy()
#y = X_full.Response
#X_full.drop(['Response'], axis=1, inplace=True)

st.title("Spot Check Models")

# Select numerical columns
#X_full.drop(['id'], axis=1, inplace=True)
#numerical_cols = [var for var in X_full.columns if X_full[var].dtype in ['float64','int64']]
#num = X_full[numerical_cols]

# Select categorical columns with relatively low cardinality (convenient but arbitrary)
#categorical_cols = [var for var in X_full.columns if
#                    X_full[var].nunique() <= 15 and 
#                    X_full[var].dtype == "object"]
# Subset with categorical features
#cat = X_full[categorical_cols]

# Numerical Features Pre-Processing
#num_o = num.copy()
# outliers correction
#for col in num_o.columns:
#    q75, q25 = np.percentile(num_o[col].dropna(), [75 ,25])
#    iqr = q75 - q25 
#    min = q25 - (iqr*1.5)
#    max = q75 + (iqr*1.5) 
#    num_o[col].mask(num_o[col]<min, min, inplace=True)
#    num_o[col].mask(num_o[col]>max, max, inplace=True)

# Categorical Features Pre-Processing
# Transform in integer binary variables
#y = y.astype('int')
#cat['Driving_License'] = cat['Driving_License'].astype('int')
#cat['Previously_Insured'] = cat['Previously_Insured'].astype('int')

#cat2 = pd.concat([y,cat], axis=1)
#cat2['Gender']=cat2['Gender'].map({'Female':0,'Male':1})
#cat2['Vehicle_Damage']=cat2['Vehicle_Damage'].map({'No':0,'Yes':1})
# calculate the mean target value per category for each feature and capture the result in a dictionary 
#Vehicle_Age_LABELS = cat2.groupby(['Vehicle_Age'])['Response'].mean().to_dict()
# replace for each feature the labels with the mean target values
#cat2['Vehicle_Age'] = cat2['Vehicle_Age'].map(Vehicle_Age_LABELS)
# Look at the new subset
#target_cat = cat2.drop(['Response'], axis=1)
# Grasp all
#X_all = pd.concat([num_o,target_cat], axis=1)

# Find features with variance equal zero 
#to_drop = [col for col in X_all.columns if np.var(X_all[col]) == 0]
# Drop features 
#X_all_v = X_all.drop(X_all[to_drop], axis=1)

# Correlation 
#corr_matrix = X_all_v.corr(method ='spearman')
# Select correlated features and removed it
# Select upper triangle of correlation matrix
#upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Find index of feature columns with correlation greater than 0.80
#to_drop = [column for column in upper.columns if any(upper[column].abs() > 0.80)]
# Drop features 
#X_all_c = X_all_v.drop(X_all_v[to_drop], axis=1)

# Normalization 
#scaling = MinMaxScaler()
# Normalization of numerical features
#num_sc = pd.DataFrame(scaling.fit_transform(X_all_c[['Age','Annual_Premium','Vintage']]), columns= ['Age','Annual_Premium','Vintage'])
# Grasp all
#X_all_sc = pd.concat([num_sc, X_all_c[['Gender','Driving_License','Previously_Insured']]], axis=1)

# Split data set
# Break off train and test set from data
#X_train, X_test, y_train, y_test = train_test_split(X_all_sc, y, train_size=0.8, test_size=0.2,stratify=y,random_state=0)

# Spot Check Algorithms
#models = []
#models.append(('LR', LogisticRegression(random_state=0)))
#models.append(('KNB', KNeighborsClassifier()))
#models.append(('GNB', GaussianNB()))
#models.append(('HGBM', HistGradientBoostingClassifier(random_state=0)))

#results_tr = []
#results_t = []
#names = []
#score = []
#skf = StratifiedKFold(n_splits=5,random_state=0, shuffle=True)
#for (name, model) in models:
#    param_grid = {}
#    my_model = GridSearchCV(model,param_grid,cv=skf)
#    my_model.fit(X_train, y_train)
#    predictions_tr = my_model.predict_proba(X_train)[:, 1]
#    predictions_t = my_model.predict_proba(X_test)[:, 1]
#    auc_train = roc_auc_score(y_train, predictions_tr)  
#    auc_test = roc_auc_score(y_test, predictions_t) 
#    results_tr.append(auc_train)
#    results_t.append(auc_test)
    
#    names.append(name)
#    f_dict = {
#        'model': name,
#        'auc_train': auc_train,
#        'auc_test': auc_test
#    }
#    score.append(f_dict)   

# Look at the auc score for each model and for each sub set
#score = pd.DataFrame(score, columns = ['model','auc_train','auc_test'])
#score

# Restore the model from file
LR_C_restored_model = pickle.load(open("https://github.com/towardsinnovationlab/Insurance_Cross_Selling_App/blob/main/LR_C_model_file.pkl","rb"))
#LR_C_restored_model = pd.read_pickle('https://github.com/towardsinnovationlab/Insurance_Cross_Selling_App/blob/main/LR_C_model_file.pkl')
predictions_tr = LR_C_restored_model.predict_proba(X_train)[:, 1]
predictions_t = LR_C_restored_model.predict_proba(X_test)[:, 1]
LR_auc_train = roc_auc_score(y_train, predictions_tr)  
LR_auc_test = roc_auc_score(y_test, predictions_t) 
score= {'model_c':['LR_C'], 'auc_train_c':[LR_auc_train],'auc_test_c':[LR_auc_test]}
LR_score= pd.DataFrame(score)
# Restore the model from file
KNB_C_restored_model = pickle.load(open("https://github.com/towardsinnovationlab/Insurance_Cross_Selling_App/blob/main/KNB_C_model_file.pkl","rb"))
#KNB_C_restored_model = pd.read_pickle('https://github.com/towardsinnovationlab/Insurance_Cross_Selling_App/blob/main/KNB_C_model_file.pkl')
predictions_tr = KNB_C_restored_model.predict_proba(X_train)[:, 1]
predictions_t = KNB_C_restored_model.predict_proba(X_test)[:, 1]
KNB_auc_train = roc_auc_score(y_train, predictions_tr)  
KNB_auc_test = roc_auc_score(y_test, predictions_t) 
score= {'model_c':['KNB_C'], 'auc_train_c':[KNB_auc_train],'auc_test_c':[KNB_auc_test]}
KNB_score= pd.DataFrame(score)
# Restore the model from file
GNB_C_restored_model = pickle.load(open("https://github.com/towardsinnovationlab/Insurance_Cross_Selling_App/blob/main/GNB_C_model_file.pkl","rb"))
#GNB_C_restored_model = pd.read_pickle('https://github.com/towardsinnovationlab/Insurance_Cross_Selling_App/blob/main/GNB_C_model_file.pkl')
predictions_tr = GNB_C_restored_model.predict_proba(X_train)[:, 1]
predictions_t = GNB_C_restored_model.predict_proba(X_test)[:, 1]
GNB_auc_train = roc_auc_score(y_train, predictions_tr)  
GNB_auc_test = roc_auc_score(y_test, predictions_t) 
score= {'model_c':['GNB_C'], 'auc_train_c':[GNB_auc_train],'auc_test_c':[GNB_auc_test]}
GNB_score= pd.DataFrame(score)
# Restore the model from file
HGBM_C_restored_model = pickle.load(open("https://github.com/towardsinnovationlab/Insurance_Cross_Selling_App/blob/main/HGBM_C_model_file.pkl","rb"))
#HGBM_C_restored_model = pd.read_pickle('https://github.com/towardsinnovationlab/Insurance_Cross_Selling_App/blob/main/HGBM_C_model_file.pkl')
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
sns.stripplot(x="model", y="auc_train",data=score,size=15)
plt.xticks(rotation=45)
plt.title('Train results')
axes = plt.gca()
axes.set_ylim([0,1.1])
plt.subplot(1,2,2)
sns.stripplot(x="model", y="auc_test",data=score,size=15)
plt.xticks(rotation=45)
plt.title('Test results')
axes = plt.gca()
axes.set_ylim([0,1.1])
st.pyplot(fig)

st.markdown("""
Logistic Regression (LR) is the benchmark model used in Insurance, and it has been compared with

Naive Bayes model (GNB), K-Nearest Neighbors model (KNB) and Hist Gradient Boosting Machine (HGBM).

""")
