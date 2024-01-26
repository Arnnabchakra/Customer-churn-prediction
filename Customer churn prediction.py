#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[50]:


df = pd.read_csv('Churn_Modelling.csv')


# In[51]:


df.head(10)


# # data preprossing

# In[52]:


df.drop(columns = ['RowNumber','CustomerId','Surname'],inplace=True)


# In[53]:


df.head(15)


# In[54]:


df.info()


# In[55]:


df.duplicated().sum()


# In[56]:


df['Exited'].value_counts()


# In[57]:


df['Geography'].value_counts()


# In[58]:


df['Gender'].value_counts()


# In[59]:


df = pd.get_dummies(df,columns=['Geography','Gender'],drop_first=True)


# In[60]:


df.head()


# In[61]:


X = df.drop(columns=['Exited'])
y = df['Exited'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[62]:


X


# In[63]:


y


# In[64]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_trf = scaler.fit_transform(X_train)
X_test_trf = scaler.transform(X_test)


#  # Ann model
# 

# In[65]:


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense


# seuential model

# In[66]:


model = Sequential()

model.add(Dense(11,activation='sigmoid',input_dim=11))
model.add(Dense(11,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))


# In[67]:


model.summary()


# In[68]:


model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[69]:


history = model.fit(X_train,y_train,batch_size=50,epochs=100,verbose=1,validation_split=0.2)


# In[70]:


model.layers[1].get_weights()


# In[72]:


y_pred = model.predict(X_test)


# In[73]:


y_pred


# In[74]:


y_pred = y_pred.argmax(axis=-1)


# In[75]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[76]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy * 100:.2f}%')


# In[ ]:


from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model('new_model.h5')  # Replace 'your_model.h5' with the actual file name


# In[78]:


import matplotlib.pyplot as plt


# In[83]:


history.history


# In[86]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


# In[87]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


# In[ ]:




