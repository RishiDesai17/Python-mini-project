from flask import Flask, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

movies_df = pd.read_csv('../imdb-5000-movie-dataset/movie_metadata.csv')
movies_df['movieIndex'] = movies_df.index
movies_df = movies_df[['movieIndex','movie_title','genres','imdb_score','movie_facebook_likes','content_rating','budget','gross']]
movies_df['genres'] = movies_df.genres.str.split('|') 
movieGenres_df = movies_df[['movieIndex','movie_title','genres']]
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        movieGenres_df.at[index, genre] = 1
movieGenres_df = movieGenres_df.fillna(0)
movieGenres_df = movieGenres_df.drop('genres',1)

@app.route('/movies', methods=['GET'])
def movies():
    return {
        "movies": movies_df[['movieIndex', 'movie_title']].to_numpy()[:100].tolist()
    }

@app.route('/recommend', methods=['POST'])
def recommendations():
    inputMovies = pd.DataFrame(request.get_json())
    userMovies = movieGenres_df[movieGenres_df['movieIndex'].isin(inputMovies['movieIndex'].tolist())]
    userGenreTable = userMovies.drop('movieIndex',1).drop('movie_title',1)
    userGenreTable = userGenreTable.reset_index(drop=True)
    userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
    genreTable = movieGenres_df.set_index(movieGenres_df['movieIndex'])
    genreTable = movieGenres_df.drop('movieIndex',1).drop('movie_title',1)
    recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
    recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
    recommendations = movies_df[movies_df['movieIndex'].isin(recommendationTable_df.head(20).keys())]
    return {
        "recommendations": recommendations['movie_title'].to_numpy().tolist()
    }

if __name__ == '__main__':
   app.run(debug=True)
