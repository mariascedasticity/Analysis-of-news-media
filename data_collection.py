#!/usr/bin/env python
# coding: utf-8

# In[1]:


#############
# Libraries #
#############

import numpy as np
import pandas as pd
import csv

import re
import requests

#Time vars manipulations
import time
import datetime
from dateutil.rrule import rrule, MONTHLY

#for JSON requests and manipulations
import requests
import pyjq
from itertools import chain
import json

#web scrapper
from bs4 import BeautifulSoup

from tqdm.notebook import tqdm

#Unable warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')


# ## Collecting data from the NT Times with JSON


#You can get your NYTimes developer key here: https://developer.nytimes.com
key = 'your_key'


# ### Single query


#Sending query to the NYT Archive API
url = 'https://api.nytimes.com/svc/archive/v1/2020/04.json?api-key='+key
request = requests.get(url)
json_df = request.json()



#Saving JSON file
with open("json_df.json", "w") as write_file:
    json.dump(json_df, write_file)



#Displaying the number of NYT articles per query=month
numb = pyjq.all('.response .docs | length', json_df)[0]
print(numb)



# Extracting information we are interested in (variables)
jq_query = f'.response .docs [] | {{n_url: .web_url, snippet: .snippet, paragraph: .lead_paragraph, mult: .multimedia[] | .url, headline: .headline .main, keyword: .keywords, date: .pub_date, doc_type: .document_type, news_desk: .news_desk, section: .section_name, subsectoin: .subsectoinName, author: .byline .original, id: ._id, word_count: .word_count}}'


# Returning a dictionary with requested data
result = pyjq.all(jq_query, json_df)


# Result (dict) to dataframe
result_df = pd.DataFrame(result)



#Working with dates
result_df['date'] = pd.to_datetime(result_df['date'])
result_df['year'] = result_df['date'].dt.year
result_df['month'] = result_df['date'].dt.month
result_df['day'] = result_df['date'].dt.day


# ### Loop for collecting data in a specified time interval



#Creating a list of (year,month,day)

#Starting date (1 Jan 2015)
start = datetime.date(2015,1,1)

#Ending date (1 Apr 2020)
end = datetime.date(2020,4,1)

#The interval between each freq iteration = monthly
dates = [(dt.year, dt.month) for dt in rrule(MONTHLY, dtstart=start, until=end)]





#Creating empty dataframe with particular columns of interest
df = pd.DataFrame(columns = ['n_url', 'snippet', 'lead_paragraph', 'image', 'headline', 'date',
       'doc_type', 'news_desk', 'section', 'author', 'id', 'word_count'])

#Loop for extracting data for specified months and years
for year, month in tqdm(dates):
    #Preventing attacks
    time.sleep(15)
    
    url = f'https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={key}'
    r = requests.get(url)
    js_data = r.json()
    
    #Displaying how many articles were published in a specified date
    numb = pyjq.all('.response .docs | length', js_data)[0]
    print(f'In {month} {year} there were published {num_docs} articles')
    
    #Extracting variables of interest
    jq_q = f'.response .docs [] | {{n_url: .web_url, snippet: .snippet, lead_paragraph: .lead_paragraph, image: .multimedia[1].url, headline: .headline .main, date: .pub_date, doc_type: .document_type, news_desk: .news_desk, section: .section_name, author: .byline, id: ._id, word_count: .word_count}}'
    out = pyjq.all(jq_q, js_data)
    
    #To dataframe + appending
    g = pd.DataFrame(out)
    df = df.append(g,  sort=False)





df.to_csv('your_data_file.csv', index=False)


# ### Web scrapping full texts of articles from NYT


df = pd.read_csv('your_data_path.csv') 

def get_full_text(df):
    
    df['full_text'] = 'NaN'
    session = requests.Session()
    
    print('Scarping articles body text...'),
    #len(df)
    for j in tqdm(range(0, len(df))):
        print(j)
        try:
            url = df['n_url'][j]
            req = session.get(url)
            soup = BeautifulSoup(req.text, 'lxml')
        except Exception as e:
            print(e)
        
        #Extracting all HTML text under tag 'p'
        tags = soup.find_all('p')
        if tags == []:
            tags = soup.find_all('p', itemprop = 'articleBody')

        # Joining HTML text
        article = ''
        for p in tags:
            article = article + ' ' + p.get_text()
            article = " ".join(article.split())

        # Text to the DataFrame
        df['full_text'][j] = article

    return df





#Run the function
get_full_text(df)



#look at one observation
df.at[0, 'full_text']



#Save the data
df.to_csv('your_data_file.csv', index=False)

