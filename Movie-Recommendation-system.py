#!/usr/bin/env python
# coding: utf-8

# # Problem Statement :-
# * Make Movie Recommendation System (Content based recommender system) .

# ### Import Library

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# #### Import csv file

# In[2]:


movies = pd.read_csv('tmdb_5000_movies 2.csv')
credits = pd.read_csv('tmdb_5000_credits 2.csv')


# In[3]:


movies.head(1)


# In[4]:


movies.shape


# In[5]:


credits.head(1)


# In[6]:


credits.shape


# ### Observation :- 
# * Here the shape of movies dataset is 4803 rows and 20 features
# * and the shape of the credits dataset is 4803 rows nd 4 features
# * In movies dataset and credit dataset two features are same 'id', and 'title'
# * So, we merge two dataset using this similer feature 'title'

# In[7]:


# merge the dataframe
movies = movies.merge(credits,on='title')


# In[8]:


movies.head(2)


# ### Observation :-
# * Here the Recommender System is Content Bsed so some features are not important for dataset like " budget, homepage, original_language, original_title, popularity, production_companies, production_countries, release_date, revenue, runtime, spoken_languages, status, tagline, vote_average, vote_count".
# * And Remaining features are " genres, id, keywords, title, overview, cast, crew" which are important for Making Recommendation System

# In[9]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[10]:


movies.head(2)


# ### Observation :-
# * After dropping some features the dataset look like the above.
# * Now Next step is merge the columns 'overview', 'genres', 'keywords', 'cast'(select only top 3 cast), and 'crew' (select only direcor of movie) in 'tag' column
# * But in 'genres', 'keywords', 'cast', and 'crew' columns the formate of data is in list of Dictionary. so in this columns perform data prepercossing.

# ### Missing value treatment

# In[11]:


movies.isnull().sum()


# In[12]:


# remove 3 rows in overview
movies.dropna(inplace=True)


# In[13]:


movies.isnull().sum()


# In[14]:


# check the duplicate values in dataset
movies.duplicated().sum()


# In[15]:


movies.iloc[0].genres


# In[16]:


# change the formate of genres column 
# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"},{"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'

# After changing the output
# ['Action','Adventure','Fantasy','SciFi']


# In[17]:


# before fromatting the genres--> it is the list of string 
# so first convert into integers and then apply the function on the genres
# so performing above step'ast' used for convert string data to integers
import ast


# In[18]:


def convert(obj):
    L = []      # create new list
    for i in ast.literal_eval(obj): # ast.literal_eval use to convert the string data to integers
        L.append(i['name']) # append the 'name' keyword data into empty list
    return L  # return L


# In[19]:


movies['genres'] = movies['genres'].apply(convert) # call the function


# In[20]:


movies.head(2)


# In[21]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[22]:


movies.head(2)


# In[23]:


# select top 3 actors in cast column
def convert3(obj):
    L = []      # create new list
    counter = 0
    for i in ast.literal_eval(obj): # ast.literal_eval use to convert the string data to integers
        if counter !=3: # this condition true upto 3 actors name
            L.append(i['name']) # append the 'name' keyword data into empty list
            counter+=1
        else:    # if counter = 4 then break the condition
            break
    return L  # return L


# In[24]:


movies['cast'] = movies['cast'].apply(convert3) # call the function


# In[25]:


movies.head(2)


# In[26]:


# if jon == 'director' so we fetch 'name'
def fetch_director(obj):
    L = []      # create new list
    for i in ast.literal_eval(obj): # ast.literal_eval use to convert the string data to integers
        if i['job'] == 'Director':
            L.append(i['name']) # append the 'name' keyword data into empty list
            break
    return L  # return L


# In[27]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[28]:


movies.head(2)


# In[29]:


# convert the overview into list
movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[30]:


movies.head(2)


# In[31]:


# remove space between the words in genres, keywords, cast and crew


# In[32]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(' ','') for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(' ','') for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(' ','') for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(' ','') for i in x])


# In[33]:


movies.head(2)


# In[34]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[35]:


movies.head(2)


# In[36]:


new_df = movies[['movie_id','title','tags']]


# In[37]:


new_df.head(2)


# In[38]:


new_df['tags'] = new_df['tags'].apply(lambda x:' '.join(x))


# In[39]:


new_df.head(2)


# In[40]:


from sklearn.feature_extraction.text import CountVectorizer


# In[41]:


cv = CountVectorizer(max_features=5000, stop_words='english')


# In[42]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[43]:


vectors


# In[44]:


vectors[0]


# In[45]:


cv.get_feature_names() # this is 5000 words we select above step


# In[46]:


from sklearn.metrics.pairwise import cosine_similarity


# In[47]:


similarity = cosine_similarity(vectors)


# In[48]:


similarity


# In[49]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key= lambda x:x[1])[1:11]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[50]:


recommend('The Fast and the Furious')


# In[51]:


import pickle


# In[52]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[53]:


pickle.dump(similarity,open('similarity.pkl','wb'))

