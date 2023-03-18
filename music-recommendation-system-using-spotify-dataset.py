#!/usr/bin/env python
# coding: utf-8
# # **Building Music Recommendation System using Spotify Dataset**
# Hello and welcome to my kernel. In this kernel, I have created Music Recommendation System using Spotify Dataset. To do this, I presented some of the visualization processes to understand data and done some EDA(Exploratory Data Analysis) so we can select features that are relevant to create a Recommendation System.

# # **Import Libraries**
import numpy as np
import pandas as pd

# import seaborn as sns
# import plotly.express as px 
# import matplotlib.pyplot as plt
# run_line_magic('matplotlib', 'inline')

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist

from collections import defaultdict
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



import warnings
warnings.filterwarnings("ignore")


# # **Read Data**
data = pd.read_csv("data.csv")
genre_data = pd.read_csv('data_by_genres.csv')
year_data = pd.read_csv('data_by_year.csv')

# # **Data Understanding by Visualization and EDA**
# # **Music Over Time**
# 
# Using the data grouped by year, we can understand how the overall sound of music has changed from 1921 to 2020.
# data['year']

# def get_decade(year):
#     period_start = int(year/10) * 10
#     decade = '{}'.format(period_start)
#     return decade

# data['decade'] = data['year'].apply(get_decade)

# # **Characteristics of Different Genres**
# 
# This dataset contains the audio features for different songs along with the audio features for different genres. We can use this information to compare different genres and understand their unique differences in sound.

# top10_genres = genre_data.nlargest(10, 'popularity')

# # **Clustering Genres with K-Means**
# 
# Here, the simple K-means clustering algorithm is used to divide the genres in this dataset into ten clusters based on the numerical audio features of each genres.

# cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=15))])
# X = genre_data.select_dtypes(np.number)
# cluster_pipeline.fit(X)
# genre_data['cluster'] = cluster_pipeline.predict(X)


# # Visualizing the Clusters with t-SNE

# tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
# genre_embedding = tsne_pipeline.fit_transform(X)
# projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
# projection['genres'] = genre_data['genres']
# projection['cluster'] = genre_data['cluster']


# # # **Clustering Songs with K-Means**
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=20, 
                                   verbose=False))
                                 ], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

# # Visualizing the Clusters with PCA

# pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
# song_embedding = pca_pipeline.fit_transform(X)
# projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
# projection['title'] = data['name']
# projection['cluster'] = data['cluster_label']

# # **Build Recommender System**
# 
# * Based on the analysis and visualizations, it’s clear that similar genres tend to have data points that are located close to each other while similar types of songs are also clustered together.
# * This observation makes perfect sense. Similar genres will sound similar and will come from similar time periods while the same can be said for songs within those genres. We can use this idea to build a recommendation system by taking the data points of the songs a user has listened to and recommending songs corresponding to nearby data points.
# * [Spotipy](https://spotipy.readthedocs.io/en/2.16.1/) is a Python client for the Spotify Web API that makes it easy for developers to fetch data and query Spotify’s catalog for songs. You have to install using `pip install spotipy`
# * After installing Spotipy, you will need to create an app on the [Spotify Developer’s page](https://developer.spotify.com/) and save your Client ID and secret key.

# In[ ]:


# get_ipython().system('pip install spotipy')


# In[26]:

client_id = '4f24e3e9bdbf40ffbda7496038221e13'
secret = 'f9ec8c5073ec422d8910b1be593657a8'

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id,
                                                           client_secret=secret))

def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


# In[27]:

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return find_song(song['name'], song['year'])
        

def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_vectors = [song_data[number_cols].values for song in song_list if get_song_data(song, spotify_data) is not None]

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict


def recommend_songs( song_list, spotify_data, n_songs=10):
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')


# In[ ]:


print(recommend_songs([{'name': 'Come As You Are', 'year':1991},
                {'name': 'Smells Like Teen Spirit', 'year': 1991},
                {'name': 'Lithium', 'year': 1992},
                {'name': 'All Apologies', 'year': 1993},
                {'name': 'Stay Away', 'year': 1993}],  data))


# * This last cell will gives you a recommendation list of songs like this,
# 
# 
# ```
# [{'name': 'Life is a Highway - From "Cars"',
#   'year': 2009,
#   'artists': "['Rascal Flatts']"},
#  {'name': 'Of Wolf And Man', 'year': 1991, 'artists': "['Metallica']"},
#  {'name': 'Somebody Like You', 'year': 2002, 'artists': "['Keith Urban']"},
#  {'name': 'Kayleigh', 'year': 1992, 'artists': "['Marillion']"},
#  {'name': 'Little Secrets', 'year': 2009, 'artists': "['Passion Pit']"},
#  {'name': 'No Excuses', 'year': 1994, 'artists': "['Alice In Chains']"},
#  {'name': 'Corazón Mágico', 'year': 1995, 'artists': "['Los Fugitivos']"},
#  {'name': 'If Today Was Your Last Day',
#   'year': 2008,
#   'artists': "['Nickelback']"},
#  {'name': "Let's Get Rocked", 'year': 1992, 'artists': "['Def Leppard']"},
#  {'name': "Breakfast At Tiffany's",
#   'year': 1995,
#   'artists': "['Deep Blue Something']"}]
# ```
# 
# 
# 
# * You can change the given songs list as per your choice.

# In[39]:


recommend_songs([{'name': 'It\'s all coming back to me', 'year': 1996}], data)


# In[ ]:




