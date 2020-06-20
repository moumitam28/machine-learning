#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
heart = pd.read_csv('heartd.csv')
# heart.info() 
# thal = 301 and ca = 299 where others = 303


# In[2]:


# heart.describe()


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
# heart.hist(bins = 50 , figsize=(20,15))


# In[4]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(heart, test_size=0.2, random_state=42)


# In[5]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index,test_index in split.split(heart,heart['sex']): # we need to select some categorcal value 
    strat_train_set = heart.loc[train_index]                   # on which the output is deeply affected 
    strat_test_set = heart.loc[test_index]


# In[6]:


heart = strat_train_set.copy() # copy of strat train in heart(original data)


# ## Looking for co-relation

# In[7]:


# corr_matrix = heart.corr()
# corr_matrix['num'].sort_values(ascending = False)


# In[8]:


# from pandas.plotting import scatter_matrix
# attributes = ['num', 'ca', 'oldpeak','age','thalach']
# scatter_matrix(heart[attributes],figsize = (12,8))


# In[9]:


# heart.plot(kind = "scatter", x= "age",y = "num", alpha = 0.9)


# In[10]:


# median = heart['thal'].median() #option 3
# heart['thal'].fillna(median)
# median1 = heart['ca'].median()
# heart['ca'].fillna(median1)


# In[11]:


heart = strat_train_set.drop('num',axis=1) 
heart_labels = strat_train_set['num'].copy()


# In[12]:


heart.describe()


# In[13]:


from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy = "median")
# imputer.fit(heart)


# In[14]:


# imputer.statistics_


# In[15]:


# X = imputer.transform(heart) # X is a numpy array 
# heart_transf = pd.DataFrame(X, columns = heart.columns)# heart_transf --> transform data set after filling the missing values
# heart_transf.describe()  


# ## Creating pipelines

# In[16]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([   # pipeline takes a series of list in it
    ('imputer',SimpleImputer(strategy="median")),  
    ('std_scaler', StandardScaler()),
])


# In[17]:


heart_tr = my_pipeline.fit_transform(heart) # heart_tr is a numpy array


# In[18]:


heart_tr.shape


# ## Selecting a desired model

# In[19]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(heart_tr,heart_labels)


# In[20]:


some_data = heart.iloc[:5]
some_labels = heart_labels.iloc[:5]


# In[21]:


prepared_data = my_pipeline.transform(some_data)  # transforming the new data to the existing data


# In[22]:


model.predict(prepared_data)


# In[23]:


list(some_labels)


# ## Evaluating the Model

# In[24]:


# from sklearn.metrics import mean_squared_error
# heart_predictions = model.predict(heart_tr)
# mse = mean_squared_error(heart_labels, heart_predictions)
# rmse = np.sqrt(mse)


# In[ ]:





# ## Using better evaluation Technique - Cross Validation

# In[25]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, heart_tr,heart_labels,scoring = "neg_mean_squared_error", cv=10 )
rmse_scores = np.sqrt(-scores)


# In[26]:


rmse_scores


# In[27]:


def print_scores(scores) : 
    print("scores are : ",scores)
    print("Mean :", scores.mean())
    print("Standard Deviation:  ", scores.std())


# In[28]:


print_scores(rmse_scores)


# In[29]:


from joblib import dump,load
dump(model , 'Heart.joblib')


# ## Testing

# In[30]:


X_test = strat_test_set.drop("num",axis=1)
Y_test = strat_test_set["num"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test,final_predictions)
final_rmse = np.sqrt(final_mse)


# In[31]:


print(final_predictions,list(Y_test))


# ## Using The Model

# In[32]:


from joblib import dump,load
model = load('Heart.joblib')


# In[33]:


# prepared_data[0]


# In[34]:


feature = np.array([[-1.18892157,  0.68313005,  0.86439266,  1.22313113,  0.00298369,
       -0.41803981, -0.9595669 ,  0.93072139, -0.70929937,  0.43238294,
       -0.97114425, -0.69625788, -0.87031794]])
model.predict(feature)


# In[ ]:




