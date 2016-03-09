
# coding: utf-8

# # IEOR 242 Assignment 06
# Classify MDA sections of 10-K reports with tf-idf and the Loughran and McDonald dictionary

# In[1]:

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# File_Sampler.py provides a helper function to read extracted MDA sections from a directory in a Pandas dataframe
from File_Sampler import get_data_frame


# In[2]:

# Path to MDA section files
REPORT_PATH = 'mdna_sections/*'

# Path to the Loughran McDonald dictionary
MASTER_DICT_PATH = '../lecture/LoughranMcDonald_MasterDictionary_2014.xlsx'

# Path to classification file
CLASSIFICATION_FILE = 'MDNA_auto_classification_week6.2.csv'

# Maximum number of features
MAX_FEATURE_COUNT = 10000

# Minimum and maximum n for n-grams
N_GRAMS_MIN = 1
N_GRAMS_MAX = 3


# ## Data Preparation
# Loading the MDA extracts and auto classification file from last week's assignment.

# In[3]:

# Load reports, the MDA sections of the reports were extracted with the functions in extract_comp_name.py
report_df = get_data_frame(REPORT_PATH, 0.75)
report_df['File Name'] = report_df['MDNA_FILE_NAMES'].map(lambda r: r.split('/')[-1].replace('mdna_', ''))
report_df.head()


# In[4]:

# Load file with manual classifications
class_df = pd.read_csv(CLASSIFICATION_FILE)
class_df.head()


# In[5]:

# Merge both dataframes
df = pd.merge(report_df, class_df, how='inner', on='File Name')
print('Total number of reports: %d' % len(df))
df.head()


# ## Calculate Weights
# TfidfVectorizer: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

# In[6]:

# Create tool to calculate tf-idf features
tf = TfidfVectorizer(analyzer='word',
                     stop_words='english',
                     max_features=MAX_FEATURE_COUNT,
                     ngram_range=(N_GRAMS_MIN, N_GRAMS_MAX))
tfidf_matrix =  tf.fit_transform(df['MDNA_TEXT_BLOB'].tolist())

feature_names = tf.get_feature_names()
print('Number of features: %d' % len(feature_names))


# In[7]:

# Create a dataframe with td-idf values for each word in columns and one row per report
tfidf_df = pd.DataFrame(tfidf_matrix.toarray())
tfidf_df.columns = [i.upper() for i in feature_names]
tfidf_df.head()


# ## Scoring with Finance Dictionary
# Loughran-McDonalds dictionary source: http://www3.nd.edu/~mcdonald/Word_Lists.html

# In[8]:

# Loading the dictionary
dict = pd.read_excel(MASTER_DICT_PATH)
dict.head()


# In[9]:

# Create a smaller dictionary that only contains the words which are used in the reports
minidict = dict[dict['Word'].isin(tfidf_df.columns)]
minidict = minidict.set_index('Word')


# In[10]:

# Clean the positive & negative columns
minidict.loc[minidict['Positive'] > 0, 'Positive'] = 1
minidict.loc[minidict['Negative'] > 0, 'Negative'] = -1
minidict.head()


# In[11]:

# Just some transformations to facilitate merging
tfidf_df = tfidf_df.T 
tfidf_df.index.name='Word'
tfidf_df.head()


# In[12]:

# Merge the dictionary with the report dataframe
senti_df = pd.merge(tfidf_df, minidict, how='inner', left_index=True, right_index=True)
senti_df.head()


# In[13]:

# Calculate sentiments for each report
for i, row in df.iterrows():
    df.loc[i, 'senti_pos'] = sum(senti_df[i] * senti_df['Positive'])
    df.loc[i, 'senti_neg'] = sum(senti_df[i] * senti_df['Negative'])
df.head()


# ## Result Validation

# In[14]:

# Takes a postitive and a negative sentiment value and returns either 'pos' or 'neg'
def senti_label(pos, neg):
    if (pos + neg) >= 0:
        return 'pos'
    else:
        return 'neg'


# In[15]:

# Calculate a vector with all predicted classifications
y_pred = df.apply(lambda row: senti_label(row['senti_pos'], row['senti_neg']), axis=1)


# In[16]:

print('Accuracy classification score: %f' % accuracy_score(df['Review'], y_pred))


# In[17]:

print('Confusion matrix:')
print(confusion_matrix(df['Review'], y_pred))


# In[18]:

print('Classification report:')
print(classification_report(df['Review'], y_pred))


# In[ ]:



