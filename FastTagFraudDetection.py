#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# In[2]:


data=pd.read_csv('FastagFraudDetection.csv')


# # Data Preparation
# ### Handling missing values

# In[3]:


data.info()


# In[4]:


data.isnull().sum()


# ### Visualization 

# In[8]:


sns.countplot(x='Fraud_indicator', data=data)
plt.show()


# In[9]:


sns.countplot(data=data, x='Vehicle_Type', hue='Fraud_indicator')
plt.show()


# In[12]:


sns.pairplot(data, vars=['Transaction_Amount', 'Amount_paid', 'Vehicle_Speed'])
plt.show()


# ### Univariate Analysis 

# In[13]:


data.describe()


# In[15]:


categorical_columns= ['Vehicle_Type', 'TollBoothID', 'Lane_Type', 'Vehicle_Dimensions']

plt.style.use('ggplot')
def autopct_fun(abs_values):
    gen = iter(abs_values)
    return lambda pct: f"{pct:.1f}%\n({next(gen)})"

fig,ax = plt.subplots(nrows = 2, ncols = 2, figsize = (25,15))
ax = ax.flat

for i,col in enumerate(categorical_columns):
    df_class = data[col].value_counts().to_frame()
    labels = df_class.index
    values = df_class.iloc[:,0].to_list()
    ax[i].pie(x = values, labels = labels, autopct=autopct_fun(values), shadow = True, textprops = {'color':'white', 'fontsize':20, 'fontweight':'bold'})
    ax[i].legend(labels)
    ax[i].set_title(col, fontsize = 20, fontweight = "bold", color = "black")
    ax[i].axis('equal')
    ax[i].legend(loc = 'best')

fig.tight_layout()
fig.show()


# ### Bivariate Analysis 

# In[44]:


correlation_matrix =data[['Timestamp','Vehicle_Type','TollBoothID','Lane_Type','Vehicle_Dimensions',
            'Transaction_Amount','Amount_paid','Geographical_Location','Vehicle_Speed','Vehicle_Plate_Number']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# ### Feature Extraction 

# In[35]:


#encoding categorical variables
categorical_columns = ['Vehicle_Type', 'FastagID', 'TollBoothID', 'Lane_Type', 'Vehicle_Dimensions', 'Geographical_Location', 'Vehicle_Plate_Number', 'Fraud_indicator']
lb= LabelEncoder()

for col in categorical_columns:
    data[col]= lb.fit_transform(data[col].astype(str))


# In[36]:


data.head(5)


# In[37]:


X= data[['Transaction_ID','Vehicle_Type','TollBoothID','Lane_Type','Vehicle_Dimensions',
        'Amount_paid','Geographical_Location','Vehicle_Speed']]
y = data['Fraud_indicator']


# In[38]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[39]:


model=RandomForestClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


# In[43]:


#evaluation metrics
metrics= [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]
results= {}

for i in metrics:
    metric_name= i.__name__
    score= i(y_test, y_pred)
    results[metric_name] = score

print(results)

