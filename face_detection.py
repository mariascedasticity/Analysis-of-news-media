#!/usr/bin/env python
# coding: utf-8

# In[1]:


#############
# Libraries #
#############

import requests
from PIL import Image
import os
import pyjq
import pandas as pd
import json
import numpy as np

from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')


# ### Single image query to Microsoft Azure
# #### (https://azure.microsoft.com/en-us/services/cognitive-services/face/)

# In[3]:


# Requesting headers set Subscription key which provides access to this API. Found in your Cognitive Services accounts.
headers = {
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': 'your_key',
}

body = dict()

# Put your link here:
# My example is from NYT article Why Are Women-Led Nations Doing Better With Covid-19? 
# https://www.nytimes.com/2020/05/15/world/coronavirus-women-leaders.html?searchResultPosition=2
body["url"] = "https://static01.nyt.com/images/2020/05/15/world/15virus-interpreter-1/merlin_172259856_80693892-5f41-415e-964b-074bb625dc0f-superJumbo.jpg?quality=90&auto=webp"
body = str(body)

# Setting parameters,more here: https://docs.microsoft.com/en-us/azure/cognitive-services/face/quickstarts/python
FaceApiDetect = 'https://westeurope.api.cognitive.microsoft.com/face/v1.0/detect?returnFaceId=true&returnFaceLandmarks=false&returnFaceAttributes=age,gender&recognitionModel=recognition_01&returnRecognitionModel=false&detectionModel=detection_01'


try:
    request = requests.post(FaceApiDetect, data=body, headers=headers) 
    print("Response:" + str(request.json()))

except Exception as e:
    print(e)


# In[175]:


# Extracting attributes gender and age
faces = request.json()
jque = f'.[] | {{FaceID: .faceId, gender: .faceAttributes .gender, age:  .faceAttributes .age}}'
f = pyjq.all(jque, faces)

# to df:
d = pd.DataFrame(f)


# ### Loop for all image URLs in dataframe

# In[1]:


# Reading your data with URLs
df = pd.read_csv('your_data.csv') 
df.index.name = 'ID'


# In[4]:


# Creating a temporary dataframe
l = pd.DataFrame()


for i in tdqm(range(0,len(df))): 
    # image_URL is a variable with values = image URLs
    k = str(df.loc[i, "image_URL"])
    request = requests.post(FaceApiDetect, headers=headers, json={"url": k})
    faces = request.json()
    
    # Extracting attributes: gender and age
    jque = f'.[] | {{FaceID: .faceId, gender: .faceAttributes .gender, age:  .faceAttributes .age}}'
    dff = pyjq.all(jque, faces)
    g = pd.DataFrame(dff)
    if g.empty == True:
        continue
    # Because we're dealing with dataframe and for each image the algorithm may detect multiple faces,
    # we need to transform the data from long format to wide.
    # Then, for each URL we'll get many attributes in a single row
    
    g['ID']=i
    g['idx'] = (g.groupby(['ID']).cumcount() + 1).astype(str)
    df = (g.pivot_table(index=['ID'], 
                      columns=['idx'], 
                      values=['age', 'gender'], 
                      aggfunc='first'))
    
    df.columns = [''.join(col) for col in df.columns]

    l = l.append(df, sort=False)
    df = df.reset_index()


# In[5]:


result = pd.merge(df, l, how='left', on='ID')
result.head() 


# In[6]:


result.to_csv('your_data_file.csv', index=False)

