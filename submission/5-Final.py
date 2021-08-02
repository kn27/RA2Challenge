#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load_ext autoreload
# %autoreload 2
# %matplotlib inline


# In[3]:


from data import ID_TO_NAME_MAP_EROSION as ID_TO_NAME_MAP, NAME_TO_ID_MAP_EROSION as NAME_TO_ID_MAP
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
import torch
import os
from PIL import Image
import numpy as np
import pandas as pd


# In[4]:


handle = '/output'
#handle = './output'


# In[5]:


import logging
logger = logging.getLogger('0')
hdlr = logging.FileHandler(handle + '/logs/5-final.log')
formatter = logging.Formatter('[%(asctime)s][%(levelname)s]   %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)


# In[6]:


PATH = './training.csv'
EROSION_SCORE = handle + '/erosion_score.csv'
NARROWING_SCORE = handle + '/narrowing_score.csv'
FINAL_OUTPUT = handle + '/predictions.csv'
logger.info (f'EROSION_SCORE:{EROSION_SCORE}, NARROWING_SCORE:{NARROWING_SCORE}, FINAL_OUTPUT:{FINAL_OUTPUT}')


# In[7]:


e = pd.read_csv(EROSION_SCORE)
j = pd.read_csv(NARROWING_SCORE)
sample = pd.read_csv(PATH)


# In[8]:


logger.info(f'Shape of erosion:{e.shape}')
logger.info(f'Shape of narrowing:{j.shape}')


# In[10]:


sample.head()


# In[11]:


e['Overall_erosion'] = e.iloc[:,1:].sum(axis = 1)
j['Overall_narrowing'] = j.iloc[:,1:].sum(axis = 1)


# In[13]:


if set(e.Patient_ID ) != set(j.Patient_ID):
    logger.warning('The patients are not matching between erosion and narrowing set')
prediction = pd.merge(left = e, right = j, on = 'Patient_ID')
prediction['Overall_Tol'] = prediction['Overall_erosion'] + prediction['Overall_narrowing']
prediction = prediction.loc[:, sample.columns]
prediction.to_csv(FINAL_OUTPUT, index = False)
logger.info(f'Wrote final output to :{FINAL_OUTPUT}')


# In[14]:


logger.info(f'Shape of all:{prediction.shape}')


# In[23]:


logger.info(prediction.columns)


# In[24]:


logger.info(prediction.Patient_ID)


# In[25]:


logger.warning(f'Number of missing per joint: {dict(prediction.isna().sum()[prediction.isna().sum()>0])}')


# In[26]:


prediction.RF_mtp_E__ip

