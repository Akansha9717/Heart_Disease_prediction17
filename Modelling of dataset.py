#!/usr/bin/env python
# coding: utf-8

# # MODELLING OF DATASET

# ###### IMPORTING REQUIREMENTS

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[4]:


from matplotlib import pyplot as plt


# In[5]:


import seaborn as sns


# In[6]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


# In[7]:


import eli5
from eli5.sklearn import PermutationImportance


# In[8]:


import shap


# In[9]:


import warnings
warnings.filterwarnings('ignore')


# In[10]:


pwd


# ###### LOADING DATASET

# In[11]:


heart = pd.read_csv('heart.csv')


# ###### RENAMING COLUMNS FOR EASY WORKING

# In[12]:


heart = heart.rename(columns={"cp": "chest_pain", "trestbps": "blood_pressure", "fbs": "blood_sugar", "ca": "vessels", "chol": "cholesterol"})


# ###### scale features

# In[15]:


standardScaler = StandardScaler()
columns_to_scale = ['age', 'blood_pressure', 'cholesterol', 'thalach', 'oldpeak']
heart[columns_to_scale] = standardScaler.fit_transform(heart[columns_to_scale])


# ###### one-hot encoding categorical features

# In[17]:


heart = pd.get_dummies(heart, columns = ['sex', 'chest_pain', 'blood_sugar', 'restecg', 'exang', 'slope', 'thal'], drop_first=True)


# ###### Separate features from target labels (healthy or sick)

# In[19]:


labels = heart['target']
features = heart.drop(['target'], axis = 1)


# ###### Split features and target labels into a training set and a test set

# In[21]:


features_train , features_test, labels_train, labels_test = train_test_split(features, labels, test_size= 0.2, random_state=42)


# # Random Forest

# ###### Find the optimal number of decision trees for the Random Forest model (from a list of options)

# In[23]:


randomForest_scores = []
trees = [10, 100, 200, 500, 1000, 1500, 2000, 5000]
for x in trees:
    randomForest = RandomForestClassifier(n_estimators = x, random_state = 1, max_depth=1)
    randomForest.fit(features_train, labels_train)
    randomForest_scores.append(randomForest.score(features_test, labels_test))
print(randomForest_scores)

sns.barplot(trees, randomForest_scores, hue=randomForest_scores, palette='Blues')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy Score')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### Find the optimal max_depth for the Random Forest model (from a list of options)

# In[25]:


randomForest_scores = []
depth = [1, 5, 10, 15]
for x in depth:
    randomForest = RandomForestClassifier(n_estimators = 1000, random_state = 42, max_depth= x)
    randomForest.fit(features_train, labels_train)
    randomForest_scores.append(randomForest.score(features_test, labels_test))
print(randomForest_scores)

sns.barplot(depth, randomForest_scores, hue=randomForest_scores, palette='Blues')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy Score')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### Instantiate model with 1000 decision trees and max depth of 1 (optimal numbers based on iterated experiments above)

# In[27]:


randomForest = RandomForestClassifier(n_estimators = 1000, random_state = 1, max_depth=1)


# ###### Train the model on features and labels training data

# In[29]:


randomForest.fit(features_train, labels_train);


# ###### Test the model on features and labels test data to assess its accuracy

# In[31]:


randomForest.score(features_test, labels_test)

score = round(randomForest.score(features_test,labels_test), 3) *100

print(f"Random Forest accuracy is {score}%")


# ###### Feature ranking

# In[33]:


perm = PermutationImportance(randomForest, random_state=42).fit(features_test, labels_test)
eli5.show_weights(perm, feature_names = features_test.columns.tolist())


# In[34]:


explainer = shap.TreeExplainer(randomForest)
shap_values = explainer.shap_values(features_test)

shap.summary_plot(shap_values[1], features_test, plot_type="bar")


# ###### confusion matrix

# In[36]:


labels_predicted = randomForest.predict(features_test)
plt.subplots(figsize=(10,5))

conf_mat = confusion_matrix(labels_test, labels_predicted)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['healthy', 'sick'], yticklabels=['healthy', 'sick'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Random Forest: Confusion Matrix')


# # K-nearest neibhouring

# In[37]:


knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(features_train, labels_train)
prediction = knn.predict(features_test)


score = round(knn.score(features_test, labels_test), 3) *100
print(f"K Nearest Neighbors accuracy is {score}%")


# ###### Find the optimal k value (from 1-20)

# In[39]:


accuracyScores = []

for x in range(1,30):
    knn2 = KNeighborsClassifier(n_neighbors = x)
    knn2.fit(features_train, labels_train)
    accuracyScores.append(knn2.score(features_test, labels_test))
    
    
sns.lineplot(range(1,30), accuracyScores)
plt.xticks(np.arange(1,30,1))
plt.xlabel("K value")
plt.ylabel("Accuracy Score")


best_k = accuracyScores.index(max(accuracyScores)) + 1
max_score = round((max(accuracyScores) * 100), 2) 

print(f"Max K Nearest Neighbors Accuracy is {max_score}%")
print(f"Best K is {best_k}")


# ###### SCORE AFTER OPTIMIZATION

# In[40]:


knn = KNeighborsClassifier(n_neighbors = 23)
knn.fit(features_train, labels_train)
prediction = knn.predict(features_test)


score = round(knn.score(features_test, labels_test), 3) *100
print(f"K Nearest Neighbors accuracy is {score}%")


# ###### Feature ranking

# In[41]:


perm = PermutationImportance(knn, random_state=1).fit(features_test, labels_test)
eli5.show_weights(perm, feature_names = features_test.columns.tolist())


# ###### confusion matrix

# In[42]:


labels_predicted = knn.predict(features_test)
plt.subplots(figsize=(10,5))

conf_mat = confusion_matrix(labels_test, labels_predicted)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['healthy', 'sick'], yticklabels=['healthy', 'sick'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('K Nearest Neighbors: Confusion Matrix')


# In[43]:


print(classification_report(labels_test, labels_predicted, target_names = ['healthy', 'sick']))


# # Logistic regressor
# 

# In[44]:


logisticRegression = LogisticRegression( solver='lbfgs')
logisticRegression.fit(features_train,labels_train)
logisticRegression.score(features_test,labels_test)


score = round(logisticRegression.score(features_test,labels_test), 3) *100
print(f"Logistic Regression accuracy is {score}%")


# ###### feature ranking

# In[45]:


perm = PermutationImportance(logisticRegression, random_state=1).fit(features_test, labels_test)
eli5.show_weights(perm, feature_names = features_test.columns.tolist())


# ###### confusion matrix

# In[46]:


labels_predicted = logisticRegression.predict(features_test)
plt.subplots(figsize=(10,5))

conf_mat = confusion_matrix(labels_test, labels_predicted)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['healthy', 'sick'], yticklabels=['healthy', 'sick'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Logistic Regression: Confusion Matrix')


# # Naive-Bayes

# In[47]:


nb = GaussianNB()
nb.fit(features_train, labels_train)
nb.score(features_test,labels_test)

score = round(nb.score(features_test,labels_test), 3) *100

print(f"Naive Bayes accuracy is {score}%")


# ###### confusion matrix

# In[48]:


labels_predicted = nb.predict(features_test)
plt.subplots(figsize=(10,5))

conf_mat = confusion_matrix(labels_test, labels_predicted)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['healthy', 'sick'], yticklabels=['healthy', 'sick'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Naive Bayes: Confusion Matrix')


# # Conclusion 

# In[70]:


heart_plot = pd.read_csv('heart.csv')
heart_plot = heart_plot.rename(columns={"cp": "chest_pain", "trestbps": "blood_pressure", "fbs": "blood_sugar", "ca": "vessels", "chol": "cholesterol"})
heart_plot['health_status'] = heart_plot['target']
heart_plot['health_status'] = ["healthy" if x == 0 else "sick" for x in heart_plot['health_status']]


# In[71]:


standardScaler = StandardScaler()
columns_to_scale = ['age', 'blood_pressure', 'cholesterol', 'thalach', 'oldpeak']
heart[columns_to_scale] = standardScaler.fit_transform(heart[columns_to_scale])


# ## Summary of dataset and objectives

# #### The 1988 Cleveland dataset contained information about 303 patients. Of these 303 patients, 165 patients exhibited the presence of heart disease, and 138 patients did not exhibit the presence of heart disease.

# In[74]:


sns.countplot(data=heart_plot, x= 'health_status')


# ##### Based on 13 features included about each patient (listed below), I attempted to model and predict the presence of heart disease in patients more broadly. I also wanted to identify which features, in particular, might be strong indicators of heart disease:

# In[76]:


# age
# sex
# chest pain type
# resting blood pressure
# serum cholestorol in mg/dl
# fasting blood sugar > 120 mg/dl
# resting electrocardiographic results
# maximum heart rate achieved
# exercise induced angina
# oldpeak = ST depression induced by exercise relative to rest
# the slope of the peak exercise ST segment
# number of major vessels colored by fluoroscopy
# thal: 3 = normal; 6 = fixed defect; 7 = reversable defect (thallium heart scan or stress test)


# ## Modeling and Predictions

# ### After experimenting with four binary classification machine learning algorithms (random forest, k-nearest neighbors, logistic regression, and Naive Bayes), the algorithms that returned the most accurate heart disease predictions were k-nearest neighbors and logistic regression. Both algorithms returned an accuracy score of 90.2%.
# 
# However, I decided to move forward with k-nearest neighbors because its precision score with healthy diagnoses was higher (0.90 vs. .87). The model returned fewer false negatives, in other words: fewer false healthy diagnoses when the patients were actually sick. For this case study, I deemed that it was more dangerous to return a false negative, because the consequence could be that a sick patient does not receive the medical treatment they need. With that said, it would be advantageous to learn more about how this prediction model might actually be used in practice and other possible consequences, which would help further inform my algorithm choice.

# ## K Nearest Neighbors

# In[77]:


knn = KNeighborsClassifier(n_neighbors = 23)
knn.fit(features_train, labels_train)
prediction = knn.predict(features_test)
score = round(knn.score(features_test, labels_test), 3) *100
print(f"K Nearest Neighbors accuracy is {score}%")


# In[78]:


labels_predicted = knn.predict(features_test)
plt.subplots(figsize=(10,5))

conf_mat = confusion_matrix(labels_test, labels_predicted)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['healthy', 'sick'], yticklabels=['healthy', 'sick'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('K Nearest Neighbors: Confusion Matrix')


# ###### As demonstrated in the confusion matrix above, the model accurately predicted 29 patients who exhibited heart disease and 26 patients who did not exhibit heart disease (out of 61 total test patients). However, the model incorrectly predicted that 3 patients exhibited heart disease when in actuality they did not and 3 patients that did not exhibit heart disease when in actuality they did.

# In[79]:


print(classification_report(labels_test, labels_predicted, target_names = ['healthy', 'sick']))


# In[80]:


precision = (round(26/29, 2))           
print(f"The precision score for a healthy diagnosis is {precision}")


# ##### Feature Importance

# ##### After successfully creating a model to predict heart disease, I next attempted to identify features that might be strong indicators of heart disease.

# In[82]:


perm = PermutationImportance(knn, random_state=42).fit(features_test, labels_test)
eli5.show_weights(perm, feature_names = features_test.columns.tolist())


# #### As determined by the permutation importance algorithm above, some of the features that would be worth further exploring as potentially strong indicators of heart disease include: the number of vessels colored by a fluoroscopy, thalach, oldpeak, blood pressure, cholesterol, and chest pain.

# ## Looking Forward

# #### To produce an even more accurate heart disease prediction model, it would be helpful to obtain a larger dataset as well as a more recent dataset, since the dataset used in this project was created in 1988. There are almost certainly medical tests and metrics developed over the last 30 years that would help further improve our identification of heart disease.

# In[ ]:




