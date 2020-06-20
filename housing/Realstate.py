#!/usr/bin/env python
# coding: utf-8

# ## Real state price predictor 

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info() #information about the data


# In[5]:


housing['CHAS'].value_counts() #counting the value 


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


# # for plotting histogram
import matplotlib.pyplot as plt
# housing.hist(bins=50,figsize=(20,20),facecolor='r')
# plt.show()


# ## train-test splitting

# In[9]:


# # for learning pursope 
import numpy as np
# def split_tarin_test(data, test_ratio): # creating train and test sets , this is present in sklearn, but we can learn things
#     np.random.seed(42) # to separate the train and test set
#     shuffled = np.random.permutation(len(data)) # randomized the data
#     test_set_size = int(len(data) * test_ratio) # getting the train data
#     test_indices = shuffled[:test_set_size]     #getting the test data
#     train_indices = shuffled[test_set_size:]    #getting data for training
#     return data.iloc[train_indices],data.iloc[test_indices] 


# In[10]:


# train_set, test_set = split_tarin_test(housing, 0.2)


# In[11]:


# print(f"Rows is train set : {len(train_set)}\n Rows is test set : {len(test_set)}") #creating the train and test set


# In[12]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows is train set : {len(train_set)}\n Rows is test set : {len(test_set)}") #creating the train and test set


# ## StratifiedShuffledSplit for the features which are most important for predictions

# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    


# In[14]:


# start_test_set['CHAS'].value_counts()


# In[15]:


# strat_train_set ['CHAS'].value_counts()


# In[16]:


housing = strat_train_set.copy()


# ## Looking for correlations 

# In[17]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending = False) # 1 = strong positive correlation ,+ve= will incrase with MEDV
                                            # -ve = will decrease with MEDV, -1 = strong negative correlation


# In[18]:


from pandas.plotting import scatter_matrix
attributes = ['MEDV', 'RM', 'ZN', 'LSTAT']
scatter_matrix(housing[attributes],figsize = (12,8))


# In[19]:


housing.plot(kind = "scatter", x= "RM",y = "MEDV", alpha = 0.9)


# ## Trying Out Attributes combination

# In[20]:


housing['TAXRM'] = housing['TAX']/housing['RM']
housing.head()


# In[21]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending = False) # 1 = strong positive correlation ,+ve= will incrase with MEDV
                                            # -ve = will decrease with MEDV, -1 = strong negative correlation


# In[22]:


housing.plot(kind = "scatter", x= "TAXRM",y = "MEDV", alpha = 0.9)


# In[23]:


housing = strat_train_set.drop('MEDV',axis=1) # we are not taking the TAXRM column as we are taking strat_train_set which is the original training table
housing_labels = strat_train_set['MEDV'].copy()


# ## Missing Attributes

# In[24]:


# To take care missing attributes we have three options :
#     1: Get rid of the missing data points id there is small no of missing values
#     2: Get rid of the the whole attribute if the relation between output label is not that good
#     3: set the values to some values (0 or mean or )


# In[25]:


# a = housing.dropna(subset = ["RM"]) #option 1 
# a.shape


# In[26]:


# housing.drop("RM",axis = 1) #option 2  axis=1 ie the column
# # original housing will remain unchanged


# In[27]:


median = housing["RM"].median() #option 3
housing['RM'].fillna(median)
# original housing will remain unchanged
housing.shape
housing.describe() # before filling the missing attributes


# In[28]:


# there is a class in sklearn which can compute median
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[29]:


imputer.statistics_


# In[30]:


X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns = housing.columns)  # housing_tr --> transform data set after filling the missing values
housing_tr.describe()  


# ## Scikit-learn Design

# Primarily three types of objects in scikit-learn
# 1. Estimators -  It estimates some parameter based on a dataset , Eg: imputer. It has a fit method and transform method. Fit method- Firts the data set and calculate internal parameters
# 
# 2. Transformers - transform method takes input and returns output based on the learnings from fit(). It has also a convenience funtion fit_tranform() , which fits and transforms.
# 
# 3. Predictors - LinerRegression model is a example, fit and predict are two common functions , it also gives us some score() function which will evaluate the prediction. Predictors will take numpy array as input

# ## Feature Scalling

# Primarily two types of feature scaling method
# 1. Min-max Scalling(Normalization):
#     (value - min ) / ( max - min )
#     sklearn provides a class called MinMaxScaler for this 
#     
# 2. Standardization:
#     (value - min)/ std
#     sklearn provieds a class called StandardScaler for this 
#     

# ## Creating Pipeline 

# In[31]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([   # pipeline takes a series of list in it
    ('imputer',SimpleImputer(strategy="median")),
#     ..... add as many as you want in your want    
    ('std_scaler', StandardScaler()),
])


# In[32]:


housing_num_tr = my_pipeline.fit_transform(housing) # housing_num_tr is a numpy array


# In[33]:


housing_num_tr.shape


# ## Selecting a decide model for realstate

# In[34]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[35]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]


# In[36]:


prepared_data = my_pipeline.transform(some_data)


# In[37]:


model.predict(prepared_data)


# In[38]:


list(some_labels)


# ## Evaluating the Model

# In[39]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[40]:


rmse


# ## Using better evaluation Technique - Cross Validation

# In[41]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10 )
rmse_scores = np.sqrt(-scores)


# In[42]:


rmse_scores


# In[43]:


def print_scores(scores) : 
    print("scores are : ",scores)
    print("Mean :", scores.mean())
    print("Standard Deviation:  ", scores.std())


# In[44]:


print_scores(rmse_scores)


# ## saving the model

# In[45]:


from joblib import dump,load
dump(model , 'DragonRealstate.joblib')


# ## Testing the model

# In[48]:


X_test = strat_test_set.drop("MEDV",axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
#print(final_predictions,list(Y_test))


# In[47]:


final_rmse


# In[49]:


prepared_data[0]


# ## Using The Model

# In[50]:


from joblib import dump,load
import numpy as np
model = load('DragonRealstate.joblib')
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.23782941, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)


# In[ ]:




