#!/usr/bin/env python
# coding: utf-8

# In[1]:


# classification
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[2]:


data=pd.read_csv('Churn_Modelling.csv')
data


# In[3]:


data.isnull().sum()


# In[4]:


x=data.iloc[:,3:-1]
x


# In[5]:


y=data['Exited']
y


# In[6]:


from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()                                  
x['Gender']=le.fit_transform(x['Gender']) 
x


# In[7]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[("encode",OneHotEncoder(drop="first",sparse=False),[1])],remainder="passthrough")
x=ct.fit_transform(x) 


# In[8]:


x


# In[9]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)


# In[10]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)
print(y_train)
print(x_test)
print(y_test)


# In[17]:


# support vector
from sklearn.svm import SVC  
classifier=SVC(C=1,kernel="poly") 
classifier.fit(x_train,y_train)


# In[18]:


y_pred=classifier.predict(x_test)
y_pred


# In[19]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[22]:


# logistic regression algorithm
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)  


# In[23]:


y_pred=classifier.predict(x_test)
y_pred


# In[24]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[25]:


# DecisionTree 
from sklearn.tree import DecisionTreeClassifier 

classifier=DecisionTreeClassifier(max_depth=10,min_samples_split=20)
classifier.fit(x_train,y_train) 


# In[26]:


y_pred=classifier.predict(x_test)
y_pred


# In[27]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[28]:


# random forest
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=50)  # n_estimators matlab kitne tree banane hai
classifier.fit(x_train,y_train)


# In[29]:


y_pred=classifier.predict(x_test)
y_pred


# In[30]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

