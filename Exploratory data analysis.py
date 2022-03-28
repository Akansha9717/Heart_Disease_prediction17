#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


pwd


# In[16]:


heart = pd.read_csv('C:\\Users\\Aakansha chaudhary\\heartdisease\\heart.csv')


# In[18]:


heart.head(5)


# In[21]:


heart = heart.rename(columns={"cp": "chest_pain", "trestbps": "blood_pressure", "fbs": "blood_sugar", "ca": "vessels", "chol": "cholesterol","target":"result"})


# In[22]:


heart.head(10)


# In[23]:


heart['health_status'] = heart['result']


# In[24]:


heart['health_status'] = ["healthy" if x == 0 else "sick" for x in heart['health_status']]


# In[25]:


heart['gender'] = heart['sex']
heart['gender'] = ['F' if x == 0 else 'M' for x in heart['gender']]


# In[26]:


heart.head()


# In[27]:


heart.tail()


# In[29]:


heart.shape


# In[30]:


heart.describe()


# In[32]:


heart.dtypes


# #Find and show duplicates

# In[34]:


heart[heart.duplicated(keep=False)]


# Drop duplicates

# In[36]:


heart = heart.drop_duplicates(keep='first')


# Find out how many people do and don't exhibit heart disease

# In[38]:


heart['health_status'].value_counts()


# In[127]:


# a plot to display the percentage of the positive and negative heart disease & check for balanced dataset.
labels = ['have disease', 'do not have disease']
values = heart['result'].value_counts().values
colors = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']
plt.pie(values, labels=labels, autopct='%1.1f%%',colors=colors)
plt.title('stats of health')
plt.show()


# In[103]:


sns.countplot(x = "result", data =heart, palette=['#432371',"#FAAE7B"])


# In[105]:


# Checking for null values in columns
heart.isnull().any()


# In[107]:


# Scatter plot for analysing the age factor
plt.scatter(x=heart.age[heart.result==1], y=heart.thalach[(heart.result==1)], c="yellow")
plt.scatter(x=heart.age[heart.result==0], y=heart.thalach[(heart.result==0)])
plt.legend(["Disease", "No Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


# # Get an overview distribution of each column

# In[83]:


#heart.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
# add bins ^^^
# heart.hist(align='left', color='b', edgecolor='red',
              
heart.hist(figsize=(20, 20), xlabelsize=8, ylabelsize=8,color='cyan',edgecolor='red')


# In[96]:


sns.color_palette("rocket", as_cmap=True)
sns.pairplot(heart, hue='health_status')


# ## Create a correlation heatmap

# In[43]:


heart.corr()


# In[77]:


f, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(heart.corr(),annot=True,cmap='PiYG',linewidths=.5)


# ### Zoom in on individual variables and correlations with target
# There are twice as many men in the data set

# In[46]:


heart['gender'].value_counts()


# In[128]:


heart.groupby('result').mean()


# In[134]:


pd.crosstab(heart.age,heart.result).plot(kind="bar",figsize=(20,6),color=['cyan','#ff9999'])
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()


# In[133]:


pd.crosstab(heart.sex,heart.result).plot(kind="bar",figsize=(15,6),color=['cyan','#ff9999' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# ### Distribution of heart disease between men and women

# In[48]:


heart.groupby(['gender', 'health_status'])['gender'].count()


# In[95]:


sns.countplot(data=heart, x='gender', hue='health_status', palette=['#432371',"#FAAE7B"])


# In[51]:


heart['sex'].corr(heart['result'])


# ### Distribution of heart disease between categories of chest pain
# 
# It seems like category 0 might be correlated with the absence of heart disease.

# In[97]:


sns.countplot(data=heart, x='chest_pain', hue='health_status', palette=['#432371',"#FAAE7B"])


# In[54]:


heart['chest_pain'].corr(heart['result'])


# In[56]:


heart['slope'].corr(heart['result'])


# ### Distribution of heart disease with thalach

# In[58]:


sns.distplot(heart['thalach'])


# In[98]:


f, ax = plt.subplots(figsize=(15, 5))
sns.countplot(data=heart, x=pd.cut(heart['thalach'], 10), hue='health_status', palette=['#432371',"#FAAE7B"])


# In[61]:


heart['thalach'].corr(heart['result'])


# ### Distribution of heart disease between categories of thal
# 
# It seems like category 2 might be correlated with heart disease

# In[99]:


sns.countplot(data=heart, x='thal', hue='health_status', palette=['#432371',"#FAAE7B"])


# In[65]:


heart['thal'].corr(heart['result'])


# ### Distribution of heart disease with oldpeak levels (binned)
# 
# It seems like 0-1 range might be correlated with heart disease Very similar distribution to vessels above

# In[100]:


#bin oldpeak with pd.cut

sns.countplot(data=heart, x=pd.cut(heart['oldpeak'], 6, labels=[0,1,2,3,4,5]), hue='health_status', palette=['#432371',"#FAAE7B"])


# In[68]:


heart['oldpeak'].corr(heart['result'])


# In[69]:


heart['oldpeak'].corr(heart['vessels'])


# In[70]:


sns.distplot(heart['vessels'])
sns.distplot(heart['oldpeak'])


# #### Distribution of age

# In[72]:


heart['age'].describe()


# In[73]:


heart['age'].mean()


# In[74]:


sns.distplot(heart['age'])


# In[75]:


sns.distplot(heart['blood_pressure'])


# In[76]:


sns.pairplot(heart , vars = ['age', 'cholesterol', 'thal', 'oldpeak'], hue='health_status')


# In[ ]:




