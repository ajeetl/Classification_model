#!/usr/bin/env python
# coding: utf-8

# ## Classification

# In[1]:


# Libraries to be used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Decision trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


# Grid search
from sklearn.model_selection import GridSearchCV

# Regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE


# In[30]:


# importing data 
patients = pd.read_csv('patients.csv')
patients.head()


# **In order to understand data we can perform descriptive analysis**

# In[31]:


# Checking data info
# Seems no missing values, only numerical variables (althoguh there are some, with only two possible values)
patients.info()


# In[32]:


# doing some basic statistics
patients.describe()


# **Plotting the data value in histogram and keepin the bins as 10.**
# 

# In[33]:


sns.histplot(patients['blood_pressure'], bins = 10)


# **Counting values of gender. 1 represents male and 0 represents female**

# In[34]:


patients['gender'].value_counts()


# In[35]:


# Not much difference in the number of patients in the two classes
patients.outcome.value_counts()


# In[36]:


# Correlation
patients.corr()


# **Output of the correlation. For example, bloodpressure and bloodsugar are positively correlated because they have positive value. In this way we can read the correlation table or data**

# In[38]:


# to distinguish properly boxplot is drawn
# For example, heart rate is clearly higher for class 1
sns.boxplot(x = 'outcome', y = 'heart_rate', data = patients)


# In[39]:


# Countplot for 0-1 variables
# For example, there is not much exercise in class 1
sns.countplot(x = "outcome", data = patients, hue = "exercise")


# **ploting age and blood pressure in scatterplot. Age is also an important factor that determines blood pressure and cholesterol.**

# In[42]:


sns.jointplot(x='age',y='blood_pressure',data=patients)


# **We can see age group above 40 has lots of marked dots. This describes that age group above 40 has blood pressure.**

# In[43]:


sns.jointplot(x='age',y='cholesterol',data=patients)


# **Same goes for cholesterol level as well. people above 40 has cholesterol.**

# **In following way we can plot whole data keeping gender as determining factor. **

# In[77]:


sns.pairplot(patients, hue= 'gender')


# In[48]:


sns.boxplot(x = 'exercise', y = 'heart_rate', data = patients)


# **Classification and Logistic Regression**

# In[57]:


patients.head()


# In[59]:


patients_tree = DecisionTreeClassifier()
X = patients[['age','gender','pain','blood_pressure','cholesterol','blood_sugar','heart_rate','exercise']]
y = patients['outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)
patients_Reg = LogisticRegression()
patients_Reg.fit(X_train,y_train)


# In[60]:


patients_Reg.coef_


# In[61]:


patients_pred = patients_Reg.predict(X_test)

# creating classification report for prediction and original test
print(confusion_matrix(y_test,patients_pred))
print(classification_report(y_test,patients_pred))
# Through the logistic prediction model less mistakes.


# **The accuracy we achieved without optimization is 83%.**

# In[62]:


# making first model with logistic regression

X = patients[['age','gender','pain','blood_pressure','cholesterol','blood_sugar','heart_rate','exercise']]
y = patients['outcome']

#Creating training set with test size 25% and random state 40.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

patient_outcome = LogisticRegression(max_iter=500, class_weight= 'balanced', C= 0.1 )

patient_outcome.fit(X_train,y_train)


# In[63]:


patients_pred = patient_outcome.predict(X_test)

# And create the classification report for prediction and original test data outcomes
print(confusion_matrix(y_test,patients_pred))

print(classification_report(y_test,patients_pred))


# In[64]:


#optimizing by using penalty function, max iteration and class weight


penalty_list = ['l1', 'l2', 'elasticnet', 'none']
for pen in penalty_list:
    try:
        patient_outcome = LogisticRegression(max_iter = 500, class_weight = 'balanced', penalty = pen)
        patient_outcome.fit(X_train,y_train)
        patients_pred = patient_outcome.predict(X_test)
        rep_outcomes = classification_report(y_test,patients_pred, output_dict = True)
        print('With penalty', pen, 'recall for default class is', rep_outcomes['1']['recall'],
              'and accuracy is', rep_outcomes['accuracy'])
    except:
        print('We cannot use', pen, 'with the current solver.')


# In[65]:


#keeping different iteration value and c value and weights

from sklearn.model_selection import GridSearchCV

# initializing the model
model = LogisticRegression()

# number of iteration
iterations = [500, 600, 700 ]

# different C values
c_values = [0.01, 0.1, 1, 10 ]

# Class weights
weights = ['balanced', {0:0.1, 1:0.9}]


grid = dict(max_iter = iterations, C = c_values, class_weight = weights)


grid_search = GridSearchCV(estimator=model, param_grid=grid, scoring='recall')

# We fit the training data

grid_result = grid_search.fit(X, y)



print("The score is", grid_result.best_score_, 'using', grid_result.best_params_)


# **Decision Tree**

# In[66]:


patient_tree = DecisionTreeClassifier(max_depth = 6, criterion = 'gini', class_weight= {0:0.1, 1:0.9}, random_state = 42)
tree_model = patient_tree.fit(X,y)
pred_tree = tree_model.predict(X)
print(confusion_matrix(y,pred_tree))
print(classification_report(y,pred_tree))


# **Decision tree is givning 84% of accuracy**

# **Bagging**

# In[67]:


patients_bag = DecisionTreeClassifier(class_weight = 'balanced', criterion= 'entropy', max_depth= 6, random_state = 42)

# creating a bagging classifier 

patient1 = BaggingClassifier(base_estimator = patients_bag, n_estimators = 100)

# fitting the training data
patient1.fit(X_train, y_train)

pred_bag = patient1.predict(X_test)
print(confusion_matrix(y_test, pred_bag))


print(classification_report(y_test, pred_bag))


# **Bagging is giving 82 % of accuracy**

# **Random forest**

# In[68]:


forest_bank = RandomForestClassifier(n_estimators=300, class_weight = 'balanced', criterion= 'entropy', max_depth= 6, 
                                       random_state = 42)

# And we fit the training data
forest_bank.fit(X_train, y_train)

# Finally look at the results

pred_forest = forest_bank.predict(X_test)
print(confusion_matrix(y_test, pred_forest))

print(classification_report(y_test, pred_forest))


# In[69]:


criterion = ['gini', 'entropy']

# Maximum depth of the tree
max_depth = [6,8,10,12]

# Class weights
weights = ['balanced', {'no':0.1, 'yes':0.9}]

# We define the grid
grid = dict(criterion = criterion, max_depth = max_depth, class_weight = weights)

patient_forest = RandomForestClassifier(random_state = 40)

grid_search = GridSearchCV(estimator=patient_forest, param_grid=grid, scoring='roc_auc')

grid_result = grid_search.fit(X, y)

# Print out the best result
print("Best result is", grid_result.best_score_, 'using', grid_result.best_params_)


# **Best result is 0.8539041205707871 using {'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 6}**

# **Finding accuracy to the above answer**

# In[70]:


patient_forest = RandomForestClassifier(n_estimators=100, class_weight = 'balanced', criterion= 'entropy', max_depth= 8, 
                                       random_state = 42)

patient_forest.fit(X_train, y_train)

pred_forest = patient_forest.predict(X_test)
print(confusion_matrix(y_test, pred_forest))

# We get slightly different results

print(classification_report(y_test, pred_forest))


# **Random Forest gave 83% of accuracy**

# The accuracy by optimizing parameter are
# 1)Decision tree = 84%
# 2)Bagging = 84%
# 3)Random forest = 83%

# **Finding accuracy by Taking four any four defining parameter and predictor**

# In[74]:


X = patients[['blood_pressure','cholesterol','blood_sugar','heart_rate']]
y = patients['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)
patients_Reg = LogisticRegression()
patients_Reg.fit(X_train,y_train)


# In[75]:


patients_pred = patients_Reg.predict(X_test)

# And create the classification report for prediction and original test data outcomes
print(confusion_matrix(y_test,patients_pred))
print(classification_report(y_test,patients_pred))
# Through the logistic prediction model less mistakes, i.e. 6. 


# **The Best accuracy we got is 70%**
