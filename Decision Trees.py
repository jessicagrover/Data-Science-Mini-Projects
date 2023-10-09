#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


DATAPATH = '/Users/jessicagrover/Downloads/breastCancer.csv'

data = pd.read_csv(DATAPATH)

data.head()


# In[13]:


# check if the data is balanced as it is a classification data
x = data['Classification']
ax = sns.countplot(x=x, data=data)


# In[14]:


def violin_plots(x, y, data):
    for i, col in enumerate(y):
        plt.figure(i)
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        ax = sns.violinplot (x=x, y=col, data=data)
        
y = data.columns[:-1]
x = data.columns[-1]

violin_plots(x,y,data)


# In[15]:


# check null values
for col in data.columns:
    print(f'{col}: {data[col].isnull().sum()}')


# ## Preprocessing

# In[16]:


# Make the target column into 0 and 1
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Classification'] = le.fit_transform(data['Classification'])

data.head()


# In[19]:


from sklearn.model_selection import train_test_split

y = data['Classification'].values.reshape(-1,1)
X = data.drop(['Classification'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=42)


# ## Baseline Decision Tree

# In[20]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

plot_confusion_matrix(clf, X_test, y_test, cmap = plt.cm.Blues)
plt.grid(False)
plt.show()


# In[22]:


from sklearn.tree import plot_tree

plot_tree(clf, max_depth=5, filled=True)


# ## Bagging
# 

# In[23]:


from sklearn.ensemble import BaggingClassifier

bagging_clf = BaggingClassifier()

bagging_clf.fit(X_train, y_train.ravel())

plot_confusion_matrix(bagging_clf, X_test, y_test, cmap=plt.cm.Blues)
plt.grid(False)
plt.show()


# ## Random Forest Classifier

# In[24]:


from sklearn.ensemble import RandomForestClassifier

random_clf = RandomForestClassifier(100)

random_clf.fit(X_train, y_train.ravel())

plot_confusion_matrix(random_clf, X_test, y_test, cmap=plt.cm.Blues)
plt.grid(False)
plt.show()


# ## Boosting

# In[25]:


from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier()

boost_clf.fit(X_train, y_train.ravel())

plot_confusion_matrix(boost_clf, X_test, y_test, cmap=plt.cm.Blues)
plt.grid(False)
plt.show()


# In[ ]:




