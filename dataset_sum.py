#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install openpyxl # The necessary python package to read excel files')

from google.colab import drive # Since this project originally was done in Colab
drive.mount('/content/drive')

import pandas as pd
df=pd.read_excel(r'/content/Inshorts Cleaned Data.xlsx')
df.head()


# In[ ]:


df= df[['Headline','Short']]

# Removing some common meaningless expressions
df=df.replace(r'&#45','-', regex=True).replace(r'&#39','', regex=True).replace(r'&amp','and', regex=True).replace(r'&#34;','', regex=True)

# For quicker result, random 20 samples were chosen, this number can be tweaked
df=df.sample(n=20)
df.reset_index(inplace = True,drop = True)

df.isnull().sum()


# In[ ]:


''' I provided the full codes for using T-5, Pegasus and Bart transformers.
    Any of the code snippets below could be activated based on the transformer to be used ''' 

get_ipython().system('pip install transformers sentencepiece')
#########################################
# T-5 in TensorFlow
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="tf",device=0)

#########################################

# Pegasus in TensorFlow
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#from transformers import pipeline
#tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
#model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
#summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="tf",device=0)


#########################################

# BART in PyTorch
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#from transformers import pipeline
#tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
#model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
#summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, min_length=5, max_length=30, device=0)


#########################################
l = []
for i in range(len(df['Short'])):
    l.append(summarizer(df['Short'][i], min_length=10, max_length=30))
df.loc[:,'News'] = l
df.head()


# In[ ]:


df['News']=df['News'].astype('string')
df.info()


# In[ ]:


df['News']=df.News.str[18:]
df['News']=df.News.str[:-2]
df.head()

