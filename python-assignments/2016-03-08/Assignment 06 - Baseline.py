
# coding: utf-8

# # IEOR 242 Assignment 06
# Baseline classification model for MDA sections of 10-K reports

# In[1]:

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[2]:

# Path to classification file
CLASSIFICATION_FILE = 'MDNA_auto_classification_week6.2.csv'


# ## Data Loading

# In[3]:

# Load file with manual classifications
class_df = pd.read_csv(CLASSIFICATION_FILE)
class_df.head()


# ## Validation

# In[4]:

print('Number of reports: %d' % len(class_df))
print('Number of positive reports: %d' % len(class_df[class_df['Review'] == 'pos']))
print('Number of negative reports: %d' % len(class_df[class_df['Review'] == 'neg']))


# In[5]:

# Baseline prediction 'neg'
y_pred = class_df.apply(lambda row: 'pos', axis=1)


# In[6]:

print('Accuracy classification score: %f' % accuracy_score(class_df['Review'], y_pred))


# In[7]:

print('Confusion matrix:')
print(confusion_matrix(class_df['Review'], y_pred))


# In[8]:

print('Classification report:')
print(classification_report(class_df['Review'], y_pred))


# In[ ]:



