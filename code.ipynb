# ###The list of libraries imported

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import string
import re
import nltk

###For dooing special functions anothers libraries are imported
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

###To display plots inline; within the notebook window
%matplotlib inline
from wordcloud import WordCloud, STOPWORDS
data=pd.read_csv("E:\covid19_tweets.csv")
data.head()
data.columns
data.shape
data.isnull().sum()
#The source of the tweet is calculated and stored in a dataframe

source_df=data['source'].value_counts().to_frame().reset_index().rename(columns={'index':'source','source':'count'})[:10]
source_df.head()
###Ploting the graph of Top 15 sources of the tweets

fig, ax = plt.subplots(figsize=(17,8))
ax.plot(source_df['source'], source_df['count'], marker='o', drawstyle="steps-post")
ax.set_title("TOP 15 SOURCES OF TWEETS",fontsize=25)
ax.set_xlabel("COUNT")
ax.set_ylabel("SOURCE")
###Finding the hashtags from the text column 

def find_hash(text):
    line=re.findall(r'(?<=#)\w+',text)
    return " ".join(line)
data['hashtags']=data['text'].apply(lambda x:find_hash(x))
###The count of the hashtags is noted and stored into a dataframe

hash_df=data['hashtags'].value_counts().to_frame().reset_index().rename(columns={'index':'hashtags','hashtags':'count'})[:10]
hastags=list(hash_df[(hash_df['hashtags'].notnull())&(hash_df['hashtags']!="")]['hashtags'])
hastags = [each_string.lower() for each_string in hastags]
data['hashtags'].isnull().sum()
hash_df=hash_df.drop(0)
##The table with the hashtags and its counts.
hash_df.head()
###Ploting the Top 10 hashtags that has been been highly tweeted during the covid pandemic 

plt.figure(figsize=(9,10))

sns.barplot(y=hash_df['hashtags'],x=hash_df['count'],data=hash_df)
plt.title("TOP 10 HASHTAGS TWEETED",fontsize=25)
plt.xlabel("Count")
plt.ylabel("Hashtags")
plt.show()
###Tweets are seperated by the user location.From which location the more tweets are being tweeted has been counted and saved 

location_df=data['user_location'].value_counts().to_frame().reset_index().rename(columns={'index':'user_location','user_location':'count'})[:10]
location_df.head()
###Plotted tweets from different locations worldwide

plt.figure(figsize=(9,10))

sns.barplot(y=location_df['user_location'],x=location_df['count'],data=location_df)
plt.title("TOP 10 USER LOCATIONS",fontsize=25)
plt.xlabel("Count")
plt.ylabel("Location")
plt.show()
###Next is the checking of whether the user is verified or not


user_df=data['user_verified'].value_counts().to_frame().reset_index().rename(columns={'index':'user_verified','user_verified':'count'})[:10]
user_df.head()
###The graph between the verified users has been plotted

plt.figure(figsize=(9,10))

sns.barplot(x='user_verified',y=user_df['count'],data=user_df)
plt.title("VERIFIED USERS COUNT",fontsize=25)
plt.xlabel("Verified users")
plt.ylabel("Count")
plt.show()
###The dates at which high peaks of this covid19 tweets are being observed and noted

data['tweet_date']=pd.to_datetime(data['date']).dt.date                  #since we only need date, time is not considered
tweet_date=data['tweet_date'].value_counts().to_frame().reset_index().rename(columns={'index':'date','tweet_date':'count'})
tweet_date['date']=pd.to_datetime(tweet_date['date'])
tweet_date=tweet_date.sort_values('date',ascending=False)
tweet_date.head(5)
###PLotting the daily tweet trend

plt.figure(figsize=(15,10))

sns.lineplot(x=tweet_date['date'],y=tweet_date['count'],data=tweet_date)
plt.title("DAILY TWEET TREND",fontsize=25)
plt.xlabel("Date")
plt.ylabel("Count")
plt.show()
###Finding the most mentioned persons or organizations

def find_at(text):
    line=re.findall(r'(?<=@)\w+',text)                  #finding the words starts with "@" symbol which normally represents the mentioning 
    return " ".join(line)
data['mention']=data['text'].apply(lambda x:find_at(x))
###The words with the "@" symbol has been retrieved and counted

word_df=data['mention'].value_counts().to_frame().reset_index().rename(columns={'index':'mention','mention':'count'})[:10]
mentions=list(word_df[(word_df['mention'].notnull())&(word_df['mention']!="")]['mention'])
mentions = [each_string.lower() for each_string in mentions]
###The table showing the most mentions of a person or organization

word_df.head()
###The top trend in the mention of person or organization is being plotted here

plt.figure(figsize=(15,10))

sns.barplot(x='mention',y='count',data=word_df)
plt.title("TOP 10 MENTIONS IN TWEET ",fontsize=25)
plt.xlabel("Words")
plt.ylabel("Count")
plt.show()
###Most words found in the comments other than hashtags and mentions are found

text_en=data['text']
text_en_lr_lc=text_en.apply(lambda x: x.lower())
text_en_lr_lc_pr= text_en_lr_lc.apply(lambda x: re.sub("([^A-Za-z \t])|(\w+:\/\/\S+)"," ", x))


##stopwords is checked in the text and avoided with some extra updates of the hashtags
stop_words = set(stopwords.words('english'))
stop_words.update(['#coronavirus', '#coronavirusoutbreak', '#coronavirusPandemic', '#covid19', 
                   '#covid_19', '#epitwitter', '#ihavecorona', 'amp','coronavirusupdates', 'coronavirus', 'covid','covid19'])

text_en_lr_lc_pr_sr = text_en_lr_lc_pr.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
text_en_lr_lc_pr_sr.head()
word_list = [word for line in text_en_lr_lc_pr_sr for word in line.split()]   #each text is splitted into single strings and stored as a list
word_list[:5]
###Word cloud of most used words in the tweets apart from the hashtags and mentions


fig, (ax2) = plt.subplots(1,1,figsize=[17, 10])
wordcloud2 = WordCloud(background_color='black',colormap="Blues", 
                        width=600,height=400).generate(" ".join(word_list))

ax2.imshow(wordcloud2,interpolation='bilinear')
ax2.axis('off')
ax2.set_title('MOST USED WORDS IN COMMENTS',fontsize=35)
nltk.download('vader_lexicon')
###Sentiment analysis of the tweets

sid = SentimentIntensityAnalyzer()
sentiment_scores = text_en_lr_lc_pr_sr.apply(lambda x: sid.polarity_scores(x))
sent_scores_df = pd.DataFrame(list(sentiment_scores))
#The sentiments of each tweets is converted into a table 
sent_scores_df['val'] = sent_scores_df['compound'].apply(lambda x: 'neutral' if x == 0 else ('positive' if x > 0 else 'negative'))
sent_scores_df.head()
###Pie chart of the sentiment analysis is shown

sent_counts_df=sent_scores_df['val'].value_counts()

plt.pie(sent_counts_df,shadow=True,startangle=100,explode=(0.1,0.1,0.1,),autopct='%1.1f%%',labels=sent_counts_df.index)
plt.title("TWEETS' SENTIMENT DISTRIBUTION \n", fontsize=16, color='Black')
plt.show()
###No. of followers were calculated with the user name

top10users = data.groupby(by=["user_name"])['user_followers'].max().sort_values(ascending=False)[:10]
top10users.to_frame().style.bar()
