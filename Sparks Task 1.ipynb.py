#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation - Data Science & Business Analytics Internship
# 
# ## TASK 1 - Prediction using Supervised Machine Learning
# 
# In this task it is required to predict the percentage of a student on the basis of number of hours studied using the Linear Regression supervised machine learning algorithm.
# 
# Steps:
# Step 1 - Importing the dataset
# Step 2 - Visualizing the dataset
# Step 3 - Data preparation
# Step 4 - Training the algorithm
# Step 5 - Visualizing the model
# Step 6 - Making predcitions
# Step 7 - Evaluating the model
# 
# ### Author: Fayeza Sifat Fatima 
# 
# ## STEP 1 - Importing the dataset
# 
# In this step, we will import the dataset through the link with the help of pandas library and then we will observe the data
# 

# In[9]:


# Importing all the required libraries

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 

# To ignore the warnings 
import warnings as wg
wg.filterwarnings("ignore")


# In[10]:


url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df = pd.read_csv(url)
    


# In[11]:


# now let's observe the dataset 
df.head()


# In[12]:


df.tail()


# In[13]:


# To find the number of columns and rows 
df.shape


# In[14]:


# To find more information about our dataset
df.info()


# In[15]:


df.describe()


# In[16]:


# now we will check if our dataset contains null or missings values  
df.isnull().sum()


# As we can see we do not have any null values in our data set so we can now move on to our next step
# 
# # STEP 2 - Visualizing the dataset
# 
# In this we will plot the dataset to check whether we can observe any relation between the two variables or not

# In[38]:


# Plotting the dataset
plt.rcParams["figure.figsize"] = [16,9]
df.plot(x='Hours', y='Scores', style='*', color='blue', markersize=10)
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.grid()
plt.show()


# From the graph above, we can observe that there is a linear relationship between "hours studied" and "percentage score". So, we can use the linear regression supervised machine model on it to predict further values.

# In[20]:


# we can also use .corr to determine the corelation between the variables 
df.corr()


# # STEP 3 - Data preparation
# 
# In this step we will divide the data into "features" (inputs) and "labels" (outputs). After that we will split the whole dataset into 2 parts - testing data and training data.

# In[21]:


df.head()


# In[22]:


# using iloc function we will divide the data 
X = df.iloc[:, :1].values  
y = df.iloc[:, 1:].values


# In[23]:


X


# In[25]:


y


# In[26]:


# Splitting data into training and testing data

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# # STEP 4 - Training the Algorithm
# 
# We have splited our data into training and testing sets, and now we will train our Model. 

# In[27]:


from sklearn.linear_model import LinearRegression  

model = LinearRegression()  
model.fit(X_train, y_train)


# # STEP 5 - Visualizing the model
# 
# After training the model, now its time to visualize it.

# In[29]:


line = model.coef_*X + model.intercept_

# Plotting for the training data
plt.rcParams["figure.figsize"] = [16,9]
plt.scatter(X_train, y_train, color='blue')
plt.plot(X, line, color='green');
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.grid()
plt.show()


# In[30]:


# Plotting for the testing data
plt.rcParams["figure.figsize"] = [16,9]
plt.scatter(X_test, y_test, color='blue')
plt.plot(X, line, color='green');
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.grid()
plt.show()


# # STEP 6 - Making Predictions
# 
# Now that we have trained our algorithm, it's time to make some predictions.

# In[31]:


print(X_test) # Testing data - In Hours
y_pred = model.predict(X_test) # Predicting the scores


# In[32]:


# Comparing Actual vs Predicted
y_test


# In[33]:


y_pred


# In[34]:


# Comparing Actual vs Predicted
comp = pd.DataFrame({ 'Actual':[y_test],'Predicted':[y_pred] })
comp


# In[35]:


# Testing with your own data

hours = 9.25
own_pred = model.predict([[hours]])
print("The predicted score if a person studies for",hours,"hours is",own_pred[0])


# Hence, it can be concluded that the predicted score if a person studies for 9.25 hours is 93.69173248737538
# 
# # STEP 7 - Evaluating the model
# In the last step, we are going to evaluate our trained model by calculating mean absolute error
# 
# 

# In[36]:


from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




