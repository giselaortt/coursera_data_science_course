
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     readonly/train.csv - the training set (all tickets issued 2004-2011)
#     readonly/test.csv - the test set (all tickets issued 2012-2016)
#     readonly/addresses.csv & readonly/latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `readonly/train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `readonly/test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32
#        
# ### Hints
# 
# * Make sure your code is working before submitting it to the autograder.
# 
# * Print out your result to see whether there is anything weird (e.g., all probabilities are the same).
# 
# * Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g., MLPClassifier) in this question. 
# 
# * Try to avoid global variables. If you have other functions besides blight_model, you should move those functions inside the scope of blight_model.
# 
# * Refer to the pinned threads in Week 4's discussion forum when there is something you could not figure it out.

# In[1]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc

train = pd.read_csv("train.csv", encoding = 'ISO-8859-1', low_memory=False )
test = pd.read_csv("test.csv", encoding = 'ISO-8859-1', low_memory=False )
#train = pd.read_csv("readonly/train.csv", encoding = 'ISO-8859-1', low_memory=False )
#test = pd.read_csv("readonly/test.csv", encoding = 'ISO-8859-1', low_memory=False )


# In[2]:

train.set_index( 'ticket_id', inplace=True )
test.set_index('ticket_id', inplace=True)


# In[3]:

#using just non-null => remove this cell to make it with null values
train = train[~np.isnan(train['compliance'])]


# In[4]:

#analisar porcentagem de nan para cada feature. apagar as que tiverem muito.
#testar performance do código apagando todas as linhas que tem nan.
#testar performance do codigo com menos features.
#tentar usar pca para diminuir as features


# In[5]:

to_drop = [
#            'violator_name','city',
#           'ticket_issued_date',
#           'violation_description',
           'payment_amount',
           'balance_due',
#           'country',
#           'violation_zip_code',
#           'grafitti_status',
#           'clean_up_cost',
#           'admin_fee',
#           'state_fee',
           'collection_status',
#           'late_fee',
           'payment_status',
           'payment_date',
           'compliance_detail',
#           'mailing_address_str_name'
#           'zip_code'
#           'disposition'
          ]


train = train.drop( to_drop, axis = 1 )

to_drop = [
#            'violator_name',
#           'city',
#           'ticket_issued_date',
           'violation_description',
#           'country',
#          'violation_zip_code',
#           'grafitti_status',
#           'mailing_address_str_name',
#           'clean_up_cost',
#           'admin_fee',
#           'state_fee',
#           'late_fee'
#           'zip_code'
#           'disposition'
          ]

#test = test.drop( to_drop, axis = 1 )


# In[7]:

to_str = train.columns[train.dtypes == object] 


# In[11]:

train.fillna(value=0, inplace=True)
test.fillna(value=0, inplace=True)


# In[13]:

#train = train.apply(lambda col:col.astype(str) if col.dtype=='object' else col)
#test = test.apply(lambda col:col.astype(str) if col.dtype=='object' else col)
train[to_str] = train[to_str].applymap(lambda col:str(col))
test[to_str] = test[to_str].applymap(lambda col:str(col))


# In[14]:

x = train.drop( 'compliance', axis = 1 )
y = train['compliance']


# In[15]:

from sklearn.preprocessing import LabelEncoder


expanded = x.append(test)

for col in test.columns:
    #train[col].fillna(0, inplace = True)
    label_encoder = LabelEncoder()
    label_encoder.fit(expanded[col].astype(str))
    #label_encoder.fit(train[col])
    x[col] = label_encoder.transform(x[col].astype(str))
    test[col] = label_encoder.transform(test[col].astype(str))
        


# In[16]:

x.head()


# In[17]:

def blight_model():
#    dt = DecisionTreeClassifier().fit(x, y)
    clf = GradientBoostingClassifier().fit(x,y)
    test['ans']=clf.predict( test )
    ans = test['ans']
    test.drop( ['ans'], axis = 1, inplace=True )
    
    return ans


# In[18]:

blight_model()

