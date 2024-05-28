import numpy as np 
import pandas as pd
import json
import ast
import warnings
warnings.filterwarnings("ignore")
import requests

data_credits = pd.read_csv('tmdb_5000_credits.csv')
data_movies = pd.read_csv('tmdb_5000_movies.csv')

data = data_credits
data = data.merge(data_movies,on="title")

data = data[['movie_id','title','overview','cast','crew','genres','keywords']]

data.dropna(inplace=True)

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

data.genres = data.genres.apply(convert)
data.keywords = data.keywords.apply(convert)

def get_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter == 5:
            break
        counter = counter + 1
        L.append(i['name'])
    return L

data.cast = data.cast.apply(get_cast)

def get_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == "Director":
            L.append(i['name'])
            break
    return L

data.crew = data.crew.apply(get_director)

data.overview = data.overview.apply(lambda x:x.split())

data['cast'] = data['cast'].apply(lambda x:[i.replace(" ","") for i in x])
data['crew'] = data['crew'].apply(lambda x:[i.replace(" ","") for i in x])
data['genres'] = data['genres'].apply(lambda x:[i.replace(" ","") for i in x])
data['keywords'] = data['keywords'].apply(lambda x:[i.replace(" ","") for i in x])

data['tags'] = data['overview']+data['genres'] + data['keywords'] + data['cast'] + data['crew']

movies = data[['movie_id','title','tags']]

movies['tags'] = movies['tags'].apply(lambda x:" ".join(x))

movies['tags'] = movies['tags'].apply(lambda x:x.lower())

import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
    
movies['tags'] = movies['tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000 , stop_words = "english")

movie_vector = cv.fit_transform(movies['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity_vector = cosine_similarity(movie_vector)

def get_movie_details(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=c1eca338703f9ad28230b56a68128529&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    overview = data['overview']
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w185" + poster_path
    rating = data['vote_average']
    date = data['release_date']

    return overview,full_path,rating,date


def recommend_movies(movie):
    index = movies[movies['title'] == movie].index[0]

    indexes = list(enumerate(similarity_vector[index]))
    sorted_indexes = sorted(indexes, key=lambda x: x[1], reverse=True)
    top_10_indices = [index for index, value in sorted_indexes[1:11]]
    movie_index,movie_value = sorted_indexes[0]
    search_movie_data = []
    movie_overview,movie_poster,movie_rating,movie_date = get_movie_details(movies.iloc[movie_index].movie_id)
    search_movie_data.append((movie,movie_overview,movie_poster,movie_rating,movie_date))

    recommended_movies = []
    for i in top_10_indices:
        movie_title = movies.iloc[i].title
        movie_overview,movie_poster,movie_rating,movie_date = get_movie_details(movies.iloc[i].movie_id)
        recommended_movies.append((movie_title,movie_overview,movie_poster,movie_rating,movie_date))

    return search_movie_data,recommended_movies