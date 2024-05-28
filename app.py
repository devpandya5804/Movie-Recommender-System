from flask import Flask, render_template, request, redirect, url_for, session
import recommendation

app = Flask(__name__)
app.secret_key = 'secretkey'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    movie_name = request.form['movie_name']
    session['movie_name'] = movie_name 
    search_movie_data,recommended_movies = recommendation.recommend_movies(movie_name)
    return render_template('recommendations.html', search_movie_data = search_movie_data, recommended_movies=recommended_movies, movie_name=movie_name)

if __name__ == '__main__':
    app.run(debug=True)