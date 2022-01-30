#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 12:48:35 2021
@author: michaelchurchcarson
"""
# =============================================================================
# Developing Classifcation model
# =============================================================================

# Load Libraries
import pandas
import numpy

# Importset data
kickstarter_df = pandas.read_excel("Kickstarter.xlsx")

### Pre-Processing

# Drop the launch_to_state_change_days column which contains missing values for every row where state != 'successful' and 
# drop the columns with dates in them because they cannot be interpreted by our model and there are numberical variables 
# in the dataset that indicat the day, month, and year of the creation, launch, and deacline of the projects.
kickstarter_df.drop('launch_to_state_change_days', axis=1, inplace=True)
kickstarter_df.drop('state_changed_at', axis=1, inplace=True) 
kickstarter_df.drop('created_at', axis=1, inplace=True)
kickstarter_df.drop('launched_at', axis=1, inplace=True)
kickstarter_df.drop('deadline', axis=1, inplace=True)

# Drop observations that have one or more missing value 
kickstarter_df = kickstarter_df.dropna()

# Remove the name and project_ID column so that our dataframe will contain all categroical and numerical values. 
# Then, add an index to the dataframe so that individual projects can still be idenified/selected.
# column to indentify the projects
kickstarter_df.drop('name', axis=1, inplace=True)
kickstarter_df.drop('project_id', axis=1, inplace=True)
kickstarter_df.reset_index()

# Remove predictors that are invalid. Predictors are invalid if they are only realized after the prediction
# is made at the moment the project is launched.
kickstarter_df.drop('pledged', axis=1, inplace=True)
kickstarter_df.drop('usd_pledged', axis=1, inplace=True)
kickstarter_df.drop('backers_count', axis=1, inplace=True)
kickstarter_df.drop('staff_pick', axis=1, inplace=True)
kickstarter_df.drop('spotlight', axis=1, inplace=True) 
kickstarter_df.drop('disable_communication', axis=1, inplace=True)
kickstarter_df.drop('state_changed_at_weekday', axis=1, inplace=True)
kickstarter_df.drop('state_changed_at_month', axis=1, inplace=True)
kickstarter_df.drop('state_changed_at_day', axis=1, inplace=True)
kickstarter_df.drop('state_changed_at_yr', axis=1, inplace=True)
kickstarter_df.drop('state_changed_at_hr', axis=1, inplace=True)

### Removal of redundent and irrelevant predictors

# Remove variable for usd exchange rate since this rate is only used to convert pledged to usd_pledged and we already removed
#both those columns.
kickstarter_df.drop('static_usd_rate', axis=1, inplace=True)

# Remove currency variable because it is duplicative of the country variable since the currency used to fund a project
# corresponds to the country the project was launched in. The correlation test below demonstrates this. It finds that 
# the level of correlation between these two variables is 0.98.

###Test the correlation between currency and country predictors
# Importset data again so that currency and country are not converted to categories in the kickstarter_df.
kickstarter_df_corr = pandas.read_excel("Kickstarter.xlsx")

# Convert currency and country predictors to categories
kickstarter_df_corr['country']=kickstarter_df_corr['country'].astype('category').cat.codes
kickstarter_df_corr['currency']=kickstarter_df_corr['currency'].astype('category').cat.codes
# Run correlation test 
correlation = kickstarter_df_corr.corr(method ='pearson')

# Remove one of the correlated predictors (currency) from  kickstarter_df
kickstarter_df.drop('currency', axis=1, inplace=True)

# Drop projects that have a state other than successful or failed because we are only interested in predicting if 
# the state variable will have a valuve of 'succesful' or 'failed'.
kickstarter_df.drop(kickstarter_df.loc[kickstarter_df['state']=='suspended'].index, inplace=True)
kickstarter_df.drop(kickstarter_df.loc[kickstarter_df['state']=='canceled'].index, inplace=True)
kickstarter_df.drop(kickstarter_df.loc[kickstarter_df['state']=='live'].index, inplace=True)

# Remove categorical variables for weekdays because the dataframe already contains the day, month, and year the projects
# were created and launched as numberical variables.
kickstarter_df.drop('deadline_weekday', axis=1, inplace=True)
kickstarter_df.drop('created_at_weekday', axis=1, inplace=True)
kickstarter_df.drop('launched_at_weekday', axis=1, inplace=True)

### Categorical variable encoding
# Dummify remaining categorical variables 
kickstarter_df1 = pandas.get_dummies(kickstarter_df, columns = ['state','country','category',])

# Use state_successful as the reference variable and remove state_failed. 
# If state_successful == 1, the project was successful. If state_successful == 0, it failed.
kickstarter_df1.drop('state_failed', axis=1, inplace=True)

### Setup the variables
# The target variable is state_successful, which is 1 if the project is successful and 0 if it failed
X = kickstarter_df1.drop(columns=['state_successful'])
y = kickstarter_df1['state_successful']

### GradientBoostingClassifier Model
from sklearn.ensemble import GradientBoostingClassifier

# Vary the minimum samples split between 2 and 10 to see which value gives the highest cross validation score
from sklearn.model_selection import cross_val_score
scores_list = []

for i in range (2,11):
    model = GradientBoostingClassifier(random_state =0,min_samples_split=i,n_estimators = 100)
    scores = cross_val_score(estimator=model, X=X, y=y, cv=5)
    print(i,':',numpy.average(scores))
    scores_list.append(numpy.average(scores))

# print the max accuracy score
max(scores_list)
# 4 min sample splits gives highest cross validation score

# Run a loop to try different splits of the data using the optimal value of 4 for min sample splits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

accuracy_score_list = []

for i in range(0,101):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = i)
    gbt = GradientBoostingClassifier(random_state = i,min_samples_split= 4,n_estimators = 100)
    model1 = gbt.fit(X_train, y_train)
    y_test_pred = model1.predict(X_test)
    print(i,':', accuracy_score(y_test, y_test_pred))
    accuracy_score_list.append(accuracy_score(y_test, y_test_pred))
    
max(accuracy_score_list)

# Build the model with optimal min sample splits of 4 and optimal split of data when random state is 45
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 45)
gbt = GradientBoostingClassifier(random_state = 45,min_samples_split= 4,n_estimators = 100)
model = gbt.fit(X_train, y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = model.predict(X_test)

# Calculate the accuracy score of the prediction
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred)

# Accuracy Score = 0.7708377744617353

#kickstarter_df.to_csv("classification.csv")

# =============================================================================
# Developing Clustering model
# =============================================================================

# Load Libraries
import pandas
import numpy

# Import dataset again
kickstarter_cluster_df = pandas.read_excel("Kickstarter.xlsx")

### Pre-Processing

# Drop projects that have a state other than successful or failed because we are only interested in predicting if 
# the state variable will have a valuve of 'succesful' or 'failed'.
kickstarter_cluster_df.drop(kickstarter_cluster_df.loc[kickstarter_cluster_df['state']=='suspended'].index, inplace=True)
kickstarter_cluster_df.drop(kickstarter_cluster_df.loc[kickstarter_cluster_df['state']=='canceled'].index, inplace=True)
kickstarter_cluster_df.drop(kickstarter_cluster_df.loc[kickstarter_cluster_df['state']=='live'].index, inplace=True)

# Drop the launch_to_state_change_days column which contains missing values for every row where state != 'successful'
kickstarter_cluster_df.drop('launch_to_state_change_days', axis=1, inplace=True)

# Drop any observations that have one or more missing value 
kickstarter_cluster_df = kickstarter_cluster_df.dropna()

# Dummify state and spotlight categorical variable 
kickstarter_cluster_df2 = pandas.get_dummies(kickstarter_cluster_df, columns = ['state', 'spotlight'])

# Use state_successful as the reference variable and remove state_failed. 
# If state_successful == 1, the project was successful. If state_successful == 0, it failed.
# Use spotlight_True as the reference variable and remove spotlight_False.
# If spotlight_True == 1, the project was spotlighted. If spotlight_True == 0, it was not.
kickstarter_cluster_df2.drop('state_failed', axis=1, inplace=True)
kickstarter_cluster_df2.drop('spotlight_False', axis=1, inplace=True)

# Create a new feature for the length of a project's funding period (difference between launch date and funding deadline in days as real number with decimals)

# Remove the'T' from the deadline and launched_at values
kickstarter_cluster_df2['deadline'] = kickstarter_cluster_df2['deadline'].str.replace('T', ' ')
kickstarter_cluster_df2['launched_at'] = kickstarter_cluster_df2['launched_at'].str.replace('T', ' ')

# Convert deadline and launched_at to datetime
kickstarter_cluster_df2["deadline"] = pandas.to_datetime(kickstarter_cluster_df2["deadline"])
kickstarter_cluster_df2["launched_at"] = pandas.to_datetime(kickstarter_cluster_df2["launched_at"])

# Subtract launched_at and deadline and add a new column to the dataframe ('Length_funding') to record this difference in days
kickstarter_cluster_df2['Length_funding'] = kickstarter_cluster_df2['deadline'] - kickstarter_cluster_df2['launched_at']
kickstarter_cluster_df2['Length_funding'] = kickstarter_cluster_df2['Length_funding']/numpy.timedelta64(1, 'D')

# Select attributes to be the predictors
X = kickstarter_cluster_df2[['state_successful','Length_funding','spotlight_True','launched_at_yr']]

# Standardize X
from sklearn.preprocessing import StandardScaler

scaler =StandardScaler()
X_std = scaler.fit_transform(X)

### K-means
from sklearn.cluster import KMeans

# Use silhouette score to decide on the number of clusters (k) to use
# Optimal K
from sklearn.metrics import silhouette_score
for i in range (2,10):    
    kmeans = KMeans(n_clusters=i)
    model_1 = kmeans.fit(X_std)
    labels = model_1.predict(X_std)
    print(i,':',numpy.average(silhouette_score(X,labels)))

# Perform K-Mean Clustering with optimal k value found above (k=3).
# Note that a random_state needs to be specified for the result to be reproducible.
kmeans = KMeans(n_clusters = 3, random_state = 5)
model0 = kmeans.fit(X_std)
labels = model0.predict(X_std)
print(model0.labels_)

# Count the number of projects in each cluster by adding the values from model0.labels_ to the kickstarter_cluster_df2 dataframe
# and counting the number of 1s and 0s.
kickstarter_cluster_df2['kmean'] = model0.labels_
kickstarter_cluster_df2['kmean'].value_counts()

# View the projects in each cluster
cluster1 = kickstarter_cluster_df2.loc[kickstarter_cluster_df2['kmean'] == 0]
cluster2 = kickstarter_cluster_df2.loc[kickstarter_cluster_df2['kmean'] == 1]
cluster3 = kickstarter_cluster_df2.loc[kickstarter_cluster_df2['kmean'] == 2]
cluster1
cluster2
cluster3

# See the centers of each cluster in terms of variables used to develop the cluster.
model0.cluster_centers_

# Create dataframe to store model.cluster_centers_ array and then export it to a csv
df_cluster_centers = pandas.DataFrame(model0.cluster_centers_, columns=['state_successful','Length_funding','spotlight_True','launched_at_yr'])
df_cluster_centers.to_csv("cluster_centers.csv")


# =======================================================================================================
# Grading 
# =======================================================================================================

# =============================================================================
# Developing Classifcation model
# =============================================================================
# Load Libraries
import pandas
import numpy

# Import Grading Data
kickstarter_grading_df = pandas.read_excel("Kickstarter-Grading.xlsx")

### Pre-Processing

# Drop the launch_to_state_change_days column which contains missing values for every row where state != 'successful' and 
# drop the columns with dates in them because they cannot be interpreted by our model and there are numberical variables 
# in the dataset that indicat the day, month, and year of the creation, launch, and deacline of the projects.
kickstarter_grading_df.drop('launch_to_state_change_days', axis=1, inplace=True)
kickstarter_grading_df.drop('state_changed_at', axis=1, inplace=True) 
kickstarter_grading_df.drop('created_at', axis=1, inplace=True)
kickstarter_grading_df.drop('launched_at', axis=1, inplace=True)
kickstarter_grading_df.drop('deadline', axis=1, inplace=True)

# Drop observations that have one or more missing value 
kickstarter_grading_df = kickstarter_grading_df.dropna()

# Remove the name and project_ID column so that our dataframe will contain all categroical and numerical values. 
# Then, add an index to the dataframe so that individual projects can still be idenified/selected.
# column to indentify the projects
kickstarter_grading_df.drop('name', axis=1, inplace=True)
kickstarter_grading_df.drop('project_id', axis=1, inplace=True)
kickstarter_grading_df.reset_index()

# Remove predictors that are invalid. Predictors are invalid if they are only realized after the prediction
# is made at the moment the project is launched.
kickstarter_grading_df.drop('pledged', axis=1, inplace=True)
kickstarter_grading_df.drop('usd_pledged', axis=1, inplace=True)
kickstarter_grading_df.drop('backers_count', axis=1, inplace=True)
kickstarter_grading_df.drop('staff_pick', axis=1, inplace=True)
kickstarter_grading_df.drop('spotlight', axis=1, inplace=True) 
kickstarter_grading_df.drop('disable_communication', axis=1, inplace=True)
kickstarter_grading_df.drop('state_changed_at_weekday', axis=1, inplace=True)
kickstarter_grading_df.drop('state_changed_at_month', axis=1, inplace=True)
kickstarter_grading_df.drop('state_changed_at_day', axis=1, inplace=True)
kickstarter_grading_df.drop('state_changed_at_yr', axis=1, inplace=True)
kickstarter_grading_df.drop('state_changed_at_hr', axis=1, inplace=True)

### Removal of redundent and irrelevant predictors

# Remove variable for usd exchange rate since this rate is only used to convert pledged to usd_pledged and we already removed
#both those columns.
kickstarter_grading_df.drop('static_usd_rate', axis=1, inplace=True)

# Remove currency variable because it is duplicative of the country variable since the currency used to fund a project
# corresponds to the country the project was launched in. The correlation test below demonstrates this. It finds that 
# the level of correlation between these two variables is 0.98.

###Test the correlation between currency and country predictors
# Importset data again so that currency and country are not converted to categories in the kickstarter_df.
kickstarter_grading_df_corr = pandas.read_excel("Kickstarter-Grading.xlsx")

# Convert currency and country predictors to categories
kickstarter_grading_df_corr['country']=kickstarter_grading_df_corr['country'].astype('category').cat.codes
kickstarter_grading_df_corr['currency']=kickstarter_grading_df_corr['currency'].astype('category').cat.codes
# Run correlation test 
correlation = kickstarter_grading_df_corr.corr(method ='pearson')

# Remove one of the correlated predictors (currency) from  kickstarter_df
kickstarter_grading_df.drop('currency', axis=1, inplace=True)

# Drop projects that have a state other than successful or failed because we are only interested in predicting if 
# the state variable will have a valuve of 'succesful' or 'failed'.
kickstarter_grading_df.drop(kickstarter_grading_df.loc[kickstarter_grading_df['state']=='suspended'].index, inplace=True)
kickstarter_grading_df.drop(kickstarter_grading_df.loc[kickstarter_grading_df['state']=='canceled'].index, inplace=True)
kickstarter_grading_df.drop(kickstarter_grading_df.loc[kickstarter_grading_df['state']=='live'].index, inplace=True)

# Remove categorical variables for weekdays because the dataframe already contains the day, month, and year the projects
# were created and launched as numberical variables.
kickstarter_grading_df.drop('deadline_weekday', axis=1, inplace=True)
kickstarter_grading_df.drop('created_at_weekday', axis=1, inplace=True)
kickstarter_grading_df.drop('launched_at_weekday', axis=1, inplace=True)

### Categorical variable encoding
# Dummify remaining categorical variables 
kickstarter_grading_df1 = pandas.get_dummies(kickstarter_grading_df, columns = ['state','country','category',])

# Use state_successful as the reference variable and remove state_failed. 
# If state_successful == 1, the project was successful. If state_successful == 0, it failed.
kickstarter_grading_df1.drop('state_failed', axis=1, inplace=True)

### Setup the variables
# The target variable is state_successful, which is 1 if the project is successful and 0 if it failed
X_grading = kickstarter_grading_df1.drop(columns=['state_successful'])
y_grading = kickstarter_grading_df1['state_successful']

# Apply the model previously trained to the grading data
y_grading_pred = model.predict(X_grading)

# Calculate the accuracy score of the prediction
from sklearn.metrics import accuracy_score
accuracy_score(y_grading, y_grading_pred)

# =============================================================================
# Developing Clustering model
# =============================================================================

# Load Libraries
import pandas
import numpy

# Import dataset again
kickstarter_grading_cluster_df = pandas.read_excel("Kickstarter-Grading.xlsx")

### Pre-Processing

# Drop projects that have a state other than successful or failed because we are only interested in predicting if 
# the state variable will have a valuve of 'succesful' or 'failed'.
kickstarter_grading_cluster_df.drop(kickstarter_grading_cluster_df.loc[kickstarter_grading_cluster_df['state']=='suspended'].index, inplace=True)
kickstarter_grading_cluster_df.drop(kickstarter_grading_cluster_df.loc[kickstarter_grading_cluster_df['state']=='canceled'].index, inplace=True)
kickstarter_grading_cluster_df.drop(kickstarter_grading_cluster_df.loc[kickstarter_grading_cluster_df['state']=='live'].index, inplace=True)

# Drop the launch_to_state_change_days column which contains missing values for every row where state != 'successful'
kickstarter_grading_cluster_df.drop('launch_to_state_change_days', axis=1, inplace=True)

# Drop any observations that have one or more missing value 
kickstarter_grading_cluster_df = kickstarter_grading_cluster_df.dropna()

# Dummify state and spotlight categorical variable 
kickstarter_grading_cluster_df2 = pandas.get_dummies(kickstarter_grading_cluster_df, columns = ['state', 'spotlight'])

# Use state_successful as the reference variable and remove state_failed. 
# If state_successful == 1, the project was successful. If state_successful == 0, it failed.
# Use spotlight_True as the reference variable and remove spotlight_False.
# If spotlight_True == 1, the project was spotlighted. If spotlight_True == 0, it was not.
kickstarter_grading_cluster_df2.drop('state_failed', axis=1, inplace=True)
kickstarter_grading_cluster_df2.drop('spotlight_False', axis=1, inplace=True)

# Create a new feature for the length of a project's funding period (difference between launch date and funding deadline in days as real number with decimals)

# Remove the'T' from the deadline and launched_at values
kickstarter_grading_cluster_df2['deadline'] = kickstarter_grading_cluster_df2['deadline'].str.replace('T', ' ')
kickstarter_grading_cluster_df2['launched_at'] = kickstarter_grading_cluster_df2['launched_at'].str.replace('T', ' ')

# Convert deadline and launched_at to datetime
kickstarter_grading_cluster_df2["deadline"] = pandas.to_datetime(kickstarter_grading_cluster_df2["deadline"])
kickstarter_grading_cluster_df2["launched_at"] = pandas.to_datetime(kickstarter_grading_cluster_df2["launched_at"])

# Subtract launched_at and deadline and add a new column to the dataframe ('Length_funding') to record this difference in days
kickstarter_grading_cluster_df2['Length_funding'] = kickstarter_grading_cluster_df2['deadline'] - kickstarter_grading_cluster_df2['launched_at']
kickstarter_grading_cluster_df2['Length_funding'] = kickstarter_grading_cluster_df2['Length_funding']/numpy.timedelta64(1, 'D')

# Select attributes to be the predictors
X_grading_cluster = kickstarter_grading_cluster_df2[['state_successful','Length_funding','spotlight_True','launched_at_yr']]

# Standardize X
from sklearn.preprocessing import StandardScaler

scaler =StandardScaler()
X_grading_std = scaler.fit_transform(X_grading_cluster)

### K-means
from sklearn.cluster import KMeans

# Use silhouette score to decide on the number of clusters (k) to use
# Optimal K
from sklearn.metrics import silhouette_score
for i in range (2,10):    
    kmeans = KMeans(n_clusters=i)
    model_1 = kmeans.fit(X_grading_std)
    labels = model_1.predict(X_grading_std)
    print(i,':',numpy.average(silhouette_score(X_grading_cluster,labels)))

# Perform K-Mean Clustering with optimal k value found above (k=3).
# Note that a random_state needs to be specified for the result to be reproducible.
kmeans = KMeans(n_clusters = 3, random_state = 5)
model0 = kmeans.fit(X_grading_std)
labels = model0.predict(X_grading_std)
print(model0.labels_)

# Count the number of projects in each cluster by adding the values from model0.labels_ to the kickstarter_cluster_df2 dataframe
# and counting the number of 1s and 0s.
kickstarter_grading_cluster_df2['kmean'] = model0.labels_
kickstarter_grading_cluster_df2['kmean'].value_counts()

# View the projects in each cluster
cluster1_grading = kickstarter_grading_cluster_df2.loc[kickstarter_grading_cluster_df2['kmean'] == 0]
cluster2_grading = kickstarter_grading_cluster_df2.loc[kickstarter_grading_cluster_df2['kmean'] == 1]
cluster3_grading = kickstarter_grading_cluster_df2.loc[kickstarter_grading_cluster_df2['kmean'] == 2]
cluster1_grading
cluster2_grading
cluster3_grading

# See the centers of each cluster in terms of variables used to develop the cluster.
model0.cluster_centers_

# Create dataframe to store model.cluster_centers_ array and then export it to a csv
df_cluster_centers = pandas.DataFrame(model0.cluster_centers_, columns=['state_successful','Length_funding','spotlight_True','launched_at_yr'])
#df_cluster_centers.to_csv("cluster_centers.csv")


### END
