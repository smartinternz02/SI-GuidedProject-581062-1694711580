#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np


# In[26]:


df=pd.read_csv("Mall_Customers.csv")
df.head()


# In[48]:


df['Gender'].value_counts()


# In[40]:


new_df = df.iloc[:,:-1]
new_df.head()


# In[51]:


#encoding gender because we cannot apply ml algorithm to string
encoding = {"Male":0, "Female":1}
new_df["Gender"] = new_df["Gender"].map(encoding)


# In[52]:


new_df.head()


# In[53]:


from sklearn import cluster


# In[54]:


error=[]
for i in range(1,5):
    kmeans = cluster.KMeans(n_clusters=i,init = 'k-means++',random_state=0)
    kmeans.fit(new_df)
    error.append(kmeans.inertia_)


# In[55]:


error


# In[58]:


km_model = cluster.KMeans(n_clusters=3,init = 'k-means++',random_state=0)


# In[59]:


km_model.fit(new_df)


# In[60]:


import matplotlib.pyplot as plt


# In[62]:


plt.plot(range(1,5),error)
plt.title('Elbow method')
plt.xlabel('number of clusters')
plt.ylabel('error')
plt.show()


# In[63]:


pred = km_model.predict(new_df)
pred


# In[66]:


# Test the model with random observation

km_model.predict([[1.6,2.9,4.3,4]])


# In[ ]:




