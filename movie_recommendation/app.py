from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Data collection and pre-processing
movies_data = pd.read_csv(r"C:\Users\prniv\OneDrive\Documents\College\ML Projects\Movie Recs\movie_recommendation\movies.csv")

# Picking relevant columns for recommendations
selected_features = ['genres', 'keywords', 'popularity', 'tagline', 'director']

# Replacing NULL values with NULL strings
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combining all the selected features
combined_features = (
    movies_data['genres'] + ' ' + 
    movies_data['keywords'] + ' ' + 
    movies_data['popularity'].astype(str) + ' ' + 
    movies_data['tagline'] + ' ' + 
    movies_data['director']
)

# Converting text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculating the cosine similarity
similarity = cosine_similarity(feature_vectors)

def get_recommendations(movie_name):
    """Get top 10 recommended movies for a given movie name."""
    all_movie_titles = movies_data['title'].tolist()
    few_close_matches = difflib.get_close_matches(movie_name, all_movie_titles)

    if not few_close_matches:
        return ["No close matches found."]
    
    close_match = few_close_matches[0]
    index = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, movie in enumerate(sorted_similar_movies[:11], start=1):
        movie_index = movie[0]
        title_from_index = movies_data[movies_data.index == movie_index]['title'].values[0]
        if title_from_index.lower() == movie_name.lower():
            continue
        recommendations.append(title_from_index)

    return recommendations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']
    recommendations = get_recommendations(movie_name)
    return render_template('results.html', movie_name=movie_name, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
