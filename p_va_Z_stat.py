#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:36:08 2021

@author: chinu
"""

#https://www.kaggle.com/jayateerthas/boston-dataset-analysis


from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy import stats

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

boston = load_boston()
columns = boston.feature_names

boston_df_x= pd.DataFrame(boston.data, columns = columns)
boston_df_x_scaler = StandardScaler().fit_transform(boston_df_x)
dataset_x = pd.DataFrame(boston_df_x_scaler, columns = columns)


boston_df_y= pd.DataFrame(boston.target, columns = ['target'])
boston_df_y_scaler = StandardScaler().fit_transform(boston_df_y)
dataset_y = pd.DataFrame(boston_df_y_scaler,columns = ['target'])

df_combine = dataset_x.join(dataset_y)

df_combine.head()

#scaler form

#check for prediction  


x_train,x_test,y_train,y_test = train_test_split(dataset_x, dataset_y, test_size = 0.3, random_state=42)

linR = LinearRegression()
linR.fit(x_train,y_train)

prediction = linR.predict(x_test)

score = linR.score(x_test,y_test)
#0.711226005748496

error = y_test-prediction

..........................................  
'''
calculate z-score of the model
'''

from scipy import stats

# now iterate over the remaining columns and create a new zscore column
z_scores = stats.zscore(df_combine)
#calculate z-scores of `df`

abs_z_scores = np.abs(z_scores)

filtered_entries = ((abs_z_scores > -3) & (abs_z_scores < 3)).all(axis=1)

new_df = df_combine[filtered_entries]
#[415 rows x 14 columns]

dataset_y1 = new_df['target']

dataset_x1 = new_df.drop(['target'], axis=1)

x_train,x_test,y_train,y_test = train_test_split(dataset_x1, dataset_y1, test_size = 0.3, random_state=42)

linR2 = LinearRegression()

linR2.fit(x_train,y_train)

prediction = linR2.predict(x_test)

score = linR2.score(x_test,y_test)

Out[94]: 0.6739782514175507


#box plot
#['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT'

.................   Visualize data with box plot ............

sns.boxplot(boston_df_x['CRIM'])
boston_df_x['CRIM'].describe()

sns.boxplot(boston_df_x['ZN'])
boston_df_x['ZN'].describe()

sns.boxplot(boston_df_x['INDUS'])
boston_df_x['INDUS'].describe()

sns.boxplot(boston_df_x['CHAS'])
boston_df_x['CHAS'].describe()

sns.boxplot(boston_df_x['NOX'])
boston_df_x['NOX'].describe()


sns.boxplot(boston_df_x['RM'])
boston_df_x['RM'].describe()


sns.boxplot(boston_df_x['AGE'])
boston_df_x['AGE'].describe()


sns.boxplot(boston_df_x['DIS'])
boston_df_x['DIS'].describe()

sns.boxplot(boston_df_x['RAD'])
boston_df_x['RAD'].describe()


sns.boxplot(boston_df_x['TAX'])
boston_df_x['TAX'].describe()


sns.boxplot(boston_df_x['PTRATIO'])
boston_df_x['PTRATIO'].describe()


sns.boxplot(boston_df_x['B'])
boston_df_x['B'].describe()


sns.boxplot(boston_df_x['LSTAT'])
boston_df_x['LSTAT'].describe()


....... Calculate IQR  score.......


boston = load_boston()
columns = boston.feature_names
df_combine = boston_df_x.join(boston_df_y)

def drop_outliers(df, field_name):
    # iqr = 1.5*(75%A - 25%A)
    iqr = 1.5 * (np.percentile(df[field_name], 75) - np.percentile(df[field_name], 25))
    # drop those fields whose column values < (25% of field value-iqr)value and > (75% of field value + iqr)value
    df.drop(df[df[field_name] > (iqr + np.percentile(df[field_name], 75))].index, inplace=True)
    df.drop(df[df[field_name] < (np.percentile(df[field_name], 25) - iqr)].index, inplace=True)
    
    
drop_outliers(df_combine, 'CRIM')

sns.boxplot(df_combine['CRIM'])
plt.title("Box plot after outlier removing")
plt.show()

drop_outliers(df_combine, 'ZN')
sns.boxplot(df_combine['ZN'])
plt.title("Box plot after outlier removing")
plt.show()   

drop_outliers(df_combine, 'INDUS')
sns.boxplot(df_combine['INDUS'])
plt.title("Box plot after outlier removing")
plt.show() 

drop_outliers(df_combine, 'CHAS')
sns.boxplot(df_combine['CHAS'])
plt.title("Box plot after outlier removing")
plt.show() 

drop_outliers(df_combine, 'NOX')
sns.boxplot(df_combine['NOX'])
plt.title("Box plot after outlier removing")
plt.show() 

drop_outliers(df_combine, 'RM')
sns.boxplot(df_combine['RM'])
plt.title("Box plot after outlier removing")
plt.show() 

drop_outliers(df_combine, 'AGE')
sns.boxplot(df_combine['AGE'])
plt.title("Box plot after outlier removing")
plt.show() 

drop_outliers(df_combine, 'DIS')
sns.boxplot(df_combine['DIS'])
plt.title("Box plot after outlier removing")
plt.show() 

drop_outliers(df_combine, 'RAD')
sns.boxplot(df_combine['RAD'])
plt.title("Box plot after outlier removing")
plt.show() 


drop_outliers(df_combine, 'TAX')
sns.boxplot(df_combine['TAX'])
plt.title("Box plot after outlier removing")
plt.show()

drop_outliers(df_combine, 'PTRATIO')
sns.boxplot(df_combine['PTRATIO'])
plt.title("Box plot after outlier removing")
plt.show()


drop_outliers(df_combine, 'B')
sns.boxplot(df_combine['B'])
plt.title("Box plot after outlier removing")
plt.show()

drop_outliers(df_combine, 'LSTAT')
sns.boxplot(df_combine['LSTAT'])
plt.title("Box plot after outlier removing")
plt.show()

drop_outliers(df_combine, 'target')
sns.boxplot(df_combine['target'])
plt.title("Box plot after outlier removing")
plt.show()

#boston_df_x.shape
#Out[135]: (205, 13)

y = df_combine['target']
X = df_combine.drop(['target'], axis=1)

x_train,x_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

linR = LinearRegression()
linR.fit(x_train,y_train)

prediction = linR.predict(x_test)

score = linR.score(x_test,y_test)

error = y_test-prediction

....DBSCAN (DENSITY-BASED SPATIAL CLUSTERING OF APPLICATIONS WITH NOISE).................



#convert Pandas dataframe columns to Nmpy array
dbscan_data = df_combine.values.astype('float32',copy = False)

dbscan_data = StandardScaler().fit_transform(dbscan_data)


model = DBSCAN(eps = 3, min_samples = 6, metric = 'euclidean').fit(dbscan_data)

#df_combine.shape
# (506, 14)
outlier_df = df_combine[model.labels_ == -1]
#(19, 14)
cluster_df = df_combine[model.labels_ != -1]
#[487 rows x 14 columns]


#Get information about clusters
print(model.labels_)

clusters = Counter(model.labels_)

print('Number of clusters={}'.format(len(clusters)))


#apply prediction model

y = cluster_df['target']
X = cluster_df.drop(['target'], axis=1)

x_train,x_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state=42)


linR = LinearRegression()
linR.fit(x_train,y_train)

prediction = linR.predict(x_test)

score = linR.score(x_test,y_test)

#0.8134127421188008
error = y_test-prediction



















