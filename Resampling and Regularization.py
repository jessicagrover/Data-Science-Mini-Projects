#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


Datapath= '/Users/jessicagrover/Downloads/Advertising.csv'

data= pd.read_csv(Datapath, index_col=0)
data.head()


# In[12]:


def scatter_plot(feature) :
    plt.figure(figsize=(10,5))
    plt.scatter(data[feature], data['sales'], c='blue')
    plt.xlabel(f'Money spent of {feature} ads ($)')
    plt.ylabel('Sales (k$)')
    plt.show()


# In[14]:


scatter_plot('TV')
scatter_plot('radio')
scatter_plot('newspaper')


# ## BASELINE MODEL
# 

# In[20]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


# In[18]:


X = data.drop(['sales'], axis=1)
y = data['sales'].values.reshape(-1,1)


# In[22]:


lin_reg = LinearRegression()

MSEs = cross_val_score(lin_reg, X, y, scoring='neg_mean_squared_error', cv=5)

mean_mse = np.mean(MSEs)

print(-mean_mse)


# ## Regularization
# 

# ### Ridge Regression

# In[24]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge


# In[25]:


ridge = Ridge()

parameters= {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(X,y)


# In[26]:


print(ridge_regressor.best_params_)
print(-ridge_regressor.best_score_)


# ### Lasso Regression

# In[29]:


from sklearn.linear_model import Lasso


# In[32]:


lasso = Lasso(tol=0.05)

parameters= {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)

lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_)
print(-lasso_regressor.best_score_)


# In[ ]:





# In[ ]:




