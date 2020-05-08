#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn import datasets 
from xgboost import XGBClassifier as xgb 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from xgboost import plot_importance


# In[23]:


digits = datasets.load_digits()
data = digits.data
target = digits.target


# In[25]:


X_train, X_test, Y_train, Y_test = train_test_split(data,
                                                    target,
                                                   test_size = 0.25,
                                                   random_state = 7)


# In[30]:


xgbtree = xgb().fit(X_train, Y_train)
Y_pred = xgbtree.predict(X_test)


# In[32]:


accuracy = accuracy_score(Y_test, Y_pred)
print(accuracy*100)


# In[35]:


fig,ax = plt.subplots(figsize=(10,15))
plot_importance(xgbtree,height=0.5,max_num_features=64,ax=ax)
plt.show()

