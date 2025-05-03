#!/usr/bin/env python
# coding: utf-8

# # Regression

#  ## Importing required libraries

# In[1]:


import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load the dataset

# In[2]:


df = pd.read_csv('housePrice.csv')
df.head()


# In[3]:


df.duplicated()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


# Categorical columns
cat_col = [col for col in df.columns if df[col].dtype == 'object']
print ('Categorical columns :', cat_col)

# Numerical colunns
num_col  = [col for col in df.columns if df[col].dtype != 'object']
print ('Numerical columns :', num_col)


# In[7]:


df[cat_col].nunique()


# In[13]:


df['Area'].unique()[:50]


# In[19]:


df1 = df.drop(columns=['Address'])

df1.shape


# In[8]:


round((df.isnull().sum()/df.shape[0])*100, 2)


# In[37]:


plt.boxplot(df1['Price(USD)'], vert=False)
plt.ylabel('Variable')
plt.xlabel('Area')
plt.show()


# In[39]:


mean = df1['Room'].mean()
std = df1['Room'].std()

lower_bound = mean - std*2
upper_bound = mean + std*2

print ('Lower Bound :', lower_bound)
print ('Upper Bound :', upper_bound)

df4 = df1[(df1['Room'] >= lower_bound) & (df1['Room'] <= upper_bound)]


# In[40]:


mean = df1['Price'].mean()
std = df1['Price'].std()

lower_bound = mean - std*2
upper_bound = mean + std*2

print ('Lower Bound :', lower_bound)
print ('Upper Bound :', upper_bound)

df4 = df1[(df1['Price'] >= lower_bound) & (df1['Price'] <= upper_bound)]


# In[47]:


X = df1[['Area', 'Room', 'Parking', 'Elevator', 'Price', 'Price(USD)']]


# In[48]:


from sklearn.preprocessing import MinMaxScaler

# initialising the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Numerical columns
num_col_ = [col for col in X.columns if X[col].dtype != 'object']
x1 = X
# learning the statistical parameters for each of the data and transforming
x1[num_col_] = scaler.fit_transform(x1[num_col_])
x1.head()


# In[ ]:




