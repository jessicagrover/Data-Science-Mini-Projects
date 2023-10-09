#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


DATAPATH = '/Users/jessicagrover/Downloads/mushrooms.csv'

data = pd.read_csv(DATAPATH)
data.head()


# In[8]:


x = data['class']
ax = sns.countplot(x=x, data=data)


# In[9]:


def plot_data(hue, data):
    for i, col in enumerate(data.columns):
        plt.figure(i)
        ax = sns.countplot(x=data[col], hue=hue, data=data)


# Preprocessing

# In[10]:


for col in data.columns:
    print(f"{col}: {data[col].isnull().sum()}")


# In[11]:


le = LabelEncoder()
data['class'] = le.fit_transform(data['class'])

data.head()


# In[12]:


encoded_data = pd.get_dummies(data)
encoded_data.head()


# Model

# In[13]:


y = data['class'].values.reshape(-1, 1)
X = encoded_data.drop(['class'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[15]:


logistic_reg = LogisticRegression()

logistic_reg.fit(X_train, y_train.ravel())

y_prob = logistic_reg.predict_proba(X_test)[:,1]
y_pred = np.where(y_prob > 0.5, 1, 0)


# In[16]:


log_confusion_matrix = confusion_matrix(y_test, y_pred)
log_confusion_matrix


# In[24]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[25]:


def plot_roc(roc_auc):
    plt.figure(figsize=(6,7))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, color='red', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.axis('tight')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    


# In[30]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[26]:


plot_roc(roc_auc)


# Linear Discriminant Analysis (LDA)

# In[31]:


lda = LinearDiscriminantAnalysis()

lda.fit(X_train, y_train.ravel())

y_prob_lda = lda.predict_proba(X_test)[:,1]
y_pred_lda = np.where(y_prob_lda > 0.5, 1, 0)


# In[32]:


lda_confusion_matrix = confusion_matrix(y_test, y_pred_lda)
lda_confusion_matrix


# In[33]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob_lda)
roc_auc_lda = auc(false_positive_rate, true_positive_rate)
roc_auc_lda


# In[34]:


plot_roc(roc_auc_lda)


# Quadratic Discriminant Analysis (QDA)

# In[35]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# In[36]:


qda = QuadraticDiscriminantAnalysis()

qda.fit(X_train, y_train.ravel())

y_prob_qda = qda.predict_proba(X_test)[:,1]
y_pred_qda = np.where(y_prob_qda > 0.5, 1, 0)


# In[37]:


qda_confusion_matrix = confusion_matrix(y_test, y_pred_qda)
qda_confusion_matrix


# In[38]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob_qda)
roc_auc_qda = auc(false_positive_rate, true_positive_rate)
roc_auc_qda


# In[39]:


plot_roc(roc_auc_qda)


# In[ ]:




