
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 1
# 
# In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data. 
# 
# Each line of the `dates.txt` file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.
# 
# The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 
# 
# Here is a list of some of the variants you might encounter in this dataset:
# * 04/20/2009; 04/20/09; 4/20/09; 4/3/09
# * Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
# * 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
# * Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
# * Feb 2009; Sep 2009; Oct 2010
# * 6/2008; 12/2009
# * 2009; 2010
# 
# Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
# * Assume all dates in xx/xx/xx format are mm/dd/yy
# * Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
# * If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
# * If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
# * Watch out for potential typos as this is a raw, real-life derived dataset.
# 
# With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.
# 
# For example if the original series was this:
# 
#     0    1999
#     1    2010
#     2    1978
#     3    2015
#     4    1985
# 
# Your function should return this:
# 
#     0    2
#     1    4
#     2    0
#     3    1
#     4    3
# 
# Your score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.
# 
# *This function should return a Series of length 500 and dtype int.*

# In[159]:


import pandas as pd

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)
        
df = pd.Series(doc)
df.head(10)


# In[160]:


backup=df.copy()


# In[161]:


months = {"Jan":1,
          "Feb":2, 
          "Mar":3, 
          "Apr":4,
          "May":5,
          "June":6,
          "July":7, 
          "Aug":8, 
          "Sept":9, 
          "Oct":10, 
          "Nov":11, 
          "Dec":12}


# In[162]:


#getting the pattern 00-00-0000 or 00-00-00
#df2 = df.str.extract( r'(?P<day>\d{1,2})-(?P<month>\d{1,2})-(?P<year>\d{2,4})'  ).dropna()


# In[163]:


#todo map the month names to their number
#df4 = df.str.extract(r'\b(?P<month>Jan|Feb|Mar|Apr|May|June|July|Aug|Sept|Oct|Nov|Dec).*\b(?P<day>\d{2}).*\b(?P<year>\d{4})').dropna()


# In[164]:


#todo map the month names to their number
#df5 = df.str.extract(r'\b(?P<day>\d{2}).*\b(?P<month>Jan|Feb|Mar|Apr|May|June|July|Aug|Sept|Oct|Nov|Dec).*\b(?P<year>\d{4})').dropna()


# In[165]:


one = df.str.extract(r'((?:\d{,2}\s)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:-|\.|\s|,)\s?\d{,2}[a-z]*(?:-|,|\s)?\s?\d{2,4})', expand = False )


# In[166]:


one.dropna().head()


# In[167]:


two = df.str.extract(r'((?:\d{1,2})(?:(?:\/|-)\d{1,2})(?:(?:\/|-)\d{2,4}))', expand = False)


# In[168]:


two.iloc[53]


# In[169]:


three = df.str.extract(r'((?:\d{1,2}(?:-|\/))?\d{4})', expand = False)


# In[170]:


four = df.str.extract(r'(\b(?:\d{4})\b)', expand = False)


# In[178]:


#dates = pd.to_datetime(one.fillna(two).fillna(three).fillna(four).replace('Decemeber','December',regex=True).replace('Janaury','January',regex=True))
dates = one.fillna(two).fillna(three).fillna(four).replace('Decemeber','December',regex=True).replace('Janaury','January',regex=True)
dates = dates.replace( "(^\d{1,2}[/|-]\d{1,2}[/|-])(\d\d$)", r"\1 19\2", regex = True ).replace("/ ", "/", regex=True).replace( "- ", "-", regex = True )
dates = pd.to_datetime( dates )


# In[179]:


#dates2.iloc[420]


# In[180]:


def date_sorter():
    return pd.Series(dates.sort_values().index)


# In[181]:


dates.iloc[53]


# In[182]:


dates.sort_values()


# In[183]:


backup.iloc[420]


# In[ ]:




