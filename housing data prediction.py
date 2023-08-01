#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


housedf=pd.read_csv('housedata.csv')


# In[3]:


print(housedf.head)


# In[7]:


housedf.info


# In[8]:


housedf.describe()


# In[10]:


housedf.columns


# In[11]:


sns.pairplot(housedf)


# In[13]:


sns.distplot(housedf['Price'])


# In[14]:


sns.heatmap(housedf.corr(), annot=True)


# In[15]:


X = housedf[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]


# In[17]:


y = housedf['Price']


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


lm = LinearRegression()


# In[22]:


lm.fit(X_train,y_train)


# In[23]:


print(lm.intercept_)


# In[24]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[25]:


predictions = lm.predict(X_test)


# In[26]:


plt.scatter(y_test,predictions)


# In[27]:


sns.distplot((y_test-predictions),bins=50);


# In[28]:


from sklearn import metrics


# In[29]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




