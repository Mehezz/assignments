#!/usr/bin/env python
# coding: utf-8

# # THE ADULT DATASET

The information is a replica of the notes for the abalone dataset from the UCI repository.

1. Title of Database: adult
2. Sources:
(a) Original owners of database (name/phone/snail address/email address)
US Census Bureau.
(b) Donor of database (name/phone/snail address/email address)
Ronny Kohavi and Barry Becker,
Data Mining and Visualization
Silicon Graphics.
e-mail: ronnyk@sgi.com
(c) Date received (databases may change over time without name change!)
05/19/96
3. Past Usage:
(a) Complete reference of article where it was described/used
@inproceedings{kohavi-nbtree,
author={Ron Kohavi},
title={Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid},
booktitle={Proceedings of the Second International Conference on Knowledge Discovery and Data Mining},
year = 1996,
pages={to appear}}
(b) Indication of what attribute(s) were being predicted
Salary greater or less than 50,000.
(b) Indication of study's results (i.e. Is it a good domain to use?)
Hard domain with a nice number of records.
The following results obtained using MLC++ with default settings
for the algorithms mentioned below.
Algorithm	Error
1	C4.5	15.54
2	C4.5-auto	14.46
3	C4.5-rules	14.94
4	Voted ID3 (0.6)	15.64
5	Voted ID3 (0.8)	16.47
6	T2	16.84
7	1R	19.54
8	NBTree	14.10
9	CN2	16.00
10	HOODG	14.82
11	FSS Naive Bayes	14.05
12	IDTM (Decision table)	14.46
13	Naive-Bayes	16.12
14	Nearest-neighbor (1)	21.42
15	Nearest-neighbor (3)	20.35
16	OC1	15.04
17	Pebls	Crashed. Unknown why (bounds WERE increased)
4. Relevant Information Paragraph:
Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

5. Number of Instances
48842 instances, mix of continuous and discrete (train=32561, test=16281)
45222 if instances with unknown values are removed (train=30162, test=15060)
Split into train-test using MLC++ GenCVFiles (2/3, 1/3 random).
6. Number of Attributes
6 continuous, 8 nominal attributes.

7. Attribute Information:
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
class: >50K, <=50K
8. Missing Attribute Values:
7% have missing values.
9. Class Distribution:
Probability for the label '>50K' : 23.93% / 24.78% (without unknowns)
Probability for the label '<=50K' : 76.07% / 75.22% (without unknowns)
10. Notes for Delve
One prototask (income) has been defined, using attributes 1-13 as inputs and income level as a binary target.
Missing values - These are confined to attributes 2 (workclass), 7 (occupation) and 14 (native-country). The prototask only uses cases with no missing values.
The income prototask comes with two priors, differing according to if attribute 4 (education) is considered to be nominal or ordinal.



# # Step:1   (i) Load the data

# In[1]:


get_ipython().system(' pip install numpy')
get_ipython().system(' pip install pandas')


# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Read csv file

df = pd.read_csv("E:/data/adult.csv")

df.head()


# In[4]:


# Number of columns and rows

print(df.shape)


# In[5]:


# Statistical data analysis

df.describe()


# In[6]:


# characterstic of data
df.info()


# ## (ii) Handling missing values

# In[7]:


# Identifying missing values

print(df.isnull())


# In[8]:


# Total number of missing values

print(df.isnull().sum())


# In[9]:


# Replacing missing values with same symbol 'n/a'

missing_values = {'n/a','-','?'}

df=pd.read_csv("E:/data/adult.csv",na_values=missing_values)

df.head


# In[10]:


# Column which contain all the missing values

df.isnull().all(axis=0)


# #### Observation  
#                - There is no such column which contain all the missing values.

# In[11]:


# Finding column which contain any missing value

df.isnull().any(axis=0)


# #### Observation
#                  - Column name navtive-country, ocupation, workclass contain some missing values.

# In[12]:


# Treatment of missing values in column

df.dropna(axis=0, how='any', inplace= True)
print(df)


# In[13]:


# Checking whether any missing value is left or not

df.isnull().any(axis=0)


# #### Observation
#                 - Now we can observe that there is no missing value left in this data.

# # Step:2 Data Preparation:

# ## (i). Removing unnecessary column

# In[14]:


# Remove column name 'fnlwgt'

df = df.drop('fnlwgt',axis=1)
print(df.info())


# ## (ii). Defining target variable

# In[15]:


df['income'].value_counts()


# ## (iii). Standardizing numerical data

# In[16]:


# Standardizing numerical columns
num = df.select_dtypes(include=['int64'])
num.head()


# In[17]:


# Correlation matrix

cor = num.corr()
cor


# In[18]:


# Correlation on a heatmap
plt.figure(figsize=(8,8))

sns.heatmap(cor , annot = True)
plt.show()


# In[19]:


print(num.shape)


# In[20]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df_num = pd.DataFrame(scaler.fit_transform(num), columns= num.columns)
df_num.head()


# ## (iv). Encoding categorical features

# In[21]:


# Defining categorical values

categorical = df.select_dtypes(include=['object'])
categorical.head()


# In[22]:


# Encoding categorical values

from sklearn.preprocessing import OneHotEncoder

encoder= OneHotEncoder(drop='first', sparse=False)

df_categorical = pd.DataFrame(encoder.fit_transform(categorical),columns = encoder.get_feature_names(categorical.columns))
df_categorical.head()


# In[23]:


# Combine both numerical and categorical data

df = pd.concat([df_num,df_categorical],axis=1)
df.head()


# ## (v). Train Test split

# In[24]:


y = df.pop("income_>50K")

x = df


# In[25]:


df.head()


# In[26]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train , y_test = train_test_split( x, y, train_size=0.8 , random_state = 100)


# In[27]:


print(x_train.shape, y_train.shape)

print(x_test.shape, y_test.shape)


# # Step:2  Training the Model

# In[28]:


# Applying appropiate algorithm for the adult data set

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(x_train , y_train)


# In[29]:


# printing coefficients (slope) and intercept

print(regressor.coef_)
print(regressor.intercept_)


# # Step:4 Predicting

# In[30]:


y_pred = regressor.predict(x_test)


# # Step:5 Evaluating

# ## (i). Accuracy Metrics

# In[31]:


from sklearn import metrics
accurate_score = metrics.accuracy_score(y_test , y_pred)
print(accurate_score)


# ## (ii). Confusion Metrics

# In[32]:


cm = metrics.confusion_matrix(y_test , y_pred)
print(cm)


# In[33]:


sns.distplot(cm)


# In[34]:


plt.figure(figsize = (3,3))
sns.heatmap(cm , annot = True)
plt.ylabel("y_test_values")
plt.xlabel("y_pred_value")
plt.title("Confusion Matrix ")
plt.show


# In[ ]:





# ## (iii). Precision matrix and Recall matrix

# In[35]:


precision_score = metrics.precision_score(y_test, y_pred)
print(precision_score)


# In[36]:


recall_score = metrics.recall_score(y_test , y_pred)
print(recall_score)


# ## (iv). Classification report

# In[37]:


class_report = metrics.classification_report(y_test , y_pred)
print(class_report)

# Number of features used = 95


# In[38]:


x_train.shape

# Number of features used = 95


# ## Step:6 RFE - Recursion Feauture Elimination

# In[39]:


from sklearn.linear_model import LogisticRegression

regressor_rfe = LogisticRegression()


# In[40]:


from sklearn.feature_selection import RFE

rfe = RFE(regressor_rfe , 45)
rfe = rfe.fit(x_train , y_train)


# In[41]:


print(rfe.support_)
print(rfe.ranking_)


# In[42]:


rfe_df = pd.DataFrame({'Columns' : x_train.columns, 'Included' : rfe.support_, 'Ranking' : rfe.ranking_})
rfe_df


# In[43]:


imp_col = x_train.columns[rfe.support_]
imp_col


# In[44]:


x_train_new = x_train[imp_col]

x_train_new.head()


# In[45]:


from sklearn.linear_model import LogisticRegression 
regressor_new = LogisticRegression()
regressor_new.fit(x_train_new , y_train)


# In[46]:


# Residual Analysis 
y_train_new = regressor_new.predict(x_train_new)

residual = y_train - y_train_new

sns.distplot(residual)


# In[47]:


x_test_new = x_test[imp_col]

y_test_pred_new= regressor_new.predict(x_test_new)


# In[48]:


rfe_df_new = pd.DataFrame({ 'Actual' : y_test, 'Predicted' : y_test_pred_new})
rfe_df_new


# In[49]:


from sklearn import metrics

print(' Accuracy = ', metrics.accuracy_score(y_test , y_pred))

print( 'Precision = ' , metrics.precision_score(y_test, y_pred))

print( 'Confusion matrix = ' ,metrics.confusion_matrix(y_test , y_pred) )

print()


# In[50]:


# Classification report after RFE 

print( metrics.classification_report(y_test , y_pred))


# In[51]:


# Residual analysis on RFE
residual_test_now = y_test - y_test_pred_new

sns.distplot(residual_test_now)


# ## Step:7 PCA - Principal Component Analysis

# In[52]:


plt.figure(figsize = (10,8))
sns.heatmap(x_train.corr())


# In[53]:


x_train.shape


# In[54]:


from sklearn.decomposition import PCA 
pca= PCA(random_state = 0)


# In[55]:


pca.fit(x_train)


# In[56]:


from sklearn.decomposition import PCA 

pca_final = PCA(n_components = 65, random_state = 0)
x_train_pca = pca_final.fit_transform(x_train)


# In[57]:


x_train_pca.shape


# In[58]:


corrmat = np.corrcoef(x_train_pca.T)
plt.figure(figsize = (10,5))

sns.heatmap(corrmat)


# In[59]:


x_test_pca = pca_final.transform(x_test)


# In[60]:


# Building final model with 65 features

from sklearn.linear_model import LogisticRegression
regressor_pca = LogisticRegression()
regressor_pca.fit(x_train_pca , y_train)


# In[61]:


print(regressor_pca.coef_)

print(regressor_pca.intercept_)


# In[62]:


y_train_pred_pca = regressor_pca.predict(x_train_pca)

residual_pca = y_train - y_train_pred_pca

sns.distplot(residual_pca)


# In[63]:


# Prediction

y_test_pred_pca = regressor_pca.predict(x_test_pca)

temp_df = pd.DataFrame({'Actual' : y_test , 'Predicted' : y_test_pred_pca})
temp_df.head()


# In[64]:


from sklearn import metrics
print(' Accuracy = ', metrics.accuracy_score(y_test , y_test_pred_pca))

print( 'Precision = ' , metrics.precision_score(y_test, y_test_pred_pca))

print( 'Confusion matrix = ' ,metrics.confusion_matrix(y_test , y_test_pred_pca) )


# In[65]:


residual_test_pca = y_test - y_test_pred_pca 

sns.distplot(residual_test_pca)


# In[66]:


print(metrics.classification_report(y_test , y_test_pred_pca))

