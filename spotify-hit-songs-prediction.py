#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Getting our data ready to be used with machine learning
# 
# Three main things we have to do:
#     1. Split the data into features and labels (usually `X` & `y`)
#     2. Filling (also called imputing) or disregarding missing values
#     3. Converting non-numerical values to numerical values (also called feature encoding)

# In[2]:

pl = pd.read_csv("./data/dataset-of-00s.csv");
pl.head()


# In[3]:

pl.dtypes


# In[4]:


# Assing
X = pl.drop("target", axis=1)
y = pl["target"]


# In[5]:


#It is already declared

# Choose the right model and hyperparameters
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

# We'll keep the default  hyperparameters
clf.get_params();


# ### 1.1 Make sure it's all numerical

# In[7]:


X.head()


# In[21]:


# Turn track, artist, uri into numbers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["track", "artist", "uri", "sections"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")

transformed_X = transformer.fit_transform(X).toarray()
transformed_X


# In[ ]:


pd.DataFrame(transformed_X)


# In[22]:


# Fit the model to the training data
np.random.seed(42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(transformed_X, 
                                                    y, 
                                                    test_size=0.2)

clf.fit(X_train, y_train)


# In[23]:


clf.score(X_test, y_test)


# In[24]:


# Make a prediction
# y_label = clf.predict(np.array([0, 2, 3, 4]))
y_preds =  clf.predict(X_test)
y_preds[:10]


# In[25]:


y_test.head(10) == y_preds[:10]


# In[26]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test, y_preds))


# In[27]:


confusion_matrix(y_test, y_preds)


# In[28]:


accuracy_score(y_test, y_preds)


# In[29]:


# Save a model and load it
import pickle

pickle.dump(clf, open("spotify-hit-songs-prediction-model.pkl", "wb"))


# In[30]:


# Improve a model
# Try different amount of n_estimators
np.random.seed(42)
for i in range(10, 100, 10):
    print(f"Trying model with {i} estimators")
    clf = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
    print(f"Model accuracy on test set: {clf.score(X_test, y_test) * 100:.2f}% \n")
