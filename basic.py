import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
# Load the movie dataset
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movie_data = movies.merge(credits, on='title')
movie_dataset = movie_data[['movie_id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew']]

# Preprocessing functions
def convert(obj):
    L = []
    for i in literal_eval(obj):
        L.append(i['name'])
    return L

def convert2(obj):
    L = []
    counter = 0
    for i in literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

def preprocess_tags(tags):
    tags = [i.replace(" ", "") for i in tags]
    return tags

# Handle NaN (float) values in 'overview' column
movie_dataset['overview'].fillna('', inplace=True)
movie_dataset['overview'] = movie_dataset['overview'].apply(lambda x: x.split())

movie_dataset['genres'] = movie_dataset['genres'].apply(convert)
movie_dataset['keywords'] = movie_dataset['keywords'].apply(convert)
movie_dataset['cast'] = movie_dataset['cast'].apply(convert2)
movie_dataset['crew'] = movie_dataset['crew'].apply(fetch_director)
movie_dataset['genres'] = movie_dataset['genres'].apply(preprocess_tags)
movie_dataset['keywords'] = movie_dataset['keywords'].apply(preprocess_tags)
movie_dataset['cast'] = movie_dataset['cast'].apply(preprocess_tags)
movie_dataset['crew'] = movie_dataset['crew'].apply(preprocess_tags)
movie_dataset['tags'] = movie_dataset['genres'] + movie_dataset['keywords'] + movie_dataset['cast'] + movie_dataset['crew']

# Create a string of tags
movie_dataset['tags'] = movie_dataset['tags'].apply(lambda x: " ".join(x))
movie_dataset['tags'] = movie_dataset['tags'].apply(lambda x: x.lower())

# Stem the tags
ps = PorterStemmer()
movie_dataset['tags'] = movie_dataset['tags'].apply(lambda x: " ".join([ps.stem(word) for word in x.split()]))

# Create the CountVectorizer and compute cosine similarity
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movie_dataset['tags']).toarray()
similarity = cosine_similarity(vectors)

# Create the main application window
root = tk.Tk()
root.title("Movie Recommendation App")
root.geometry("400x200")  # Set the window size

# Create a label (title)
title_label = tk.Label(root, text="Movie Recommendation App")
title_label.pack(pady=10)

# Create a combo box (select box) and populate it with movie titles
movie_titles = movie_dataset['title'].tolist()
combo_box = ttk.Combobox(root, values=movie_titles)
combo_box.pack(pady=10)

# Function to recommend movies
def recommend_movies():
    selected_movie = combo_box.get()
    if selected_movie:
        movie_index = movie_dataset[movie_dataset['title'] == selected_movie].index[0]
        distances = similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        recommended_movies = [movie_dataset.iloc[i[0]].title for i in movie_list]
        recommendations_label.config(text=f"Recommended Movies for '{selected_movie}':\n{', '.join(recommended_movies)}")
    else:
        messagebox.showerror("Error", "Please select a movie.")

# Create a button
recommend_button = tk.Button(root, text="Recommend Movies", command=recommend_movies)
recommend_button.pack(pady=10)

# Create a label to display recommendations
recommendations_label = tk.Label(root, text="", wraplength=350)
recommendations_label.pack(pady=10)

# Change the background color
root.configure(bg='lightblue')

# Start the main event loop
root.mainloop()
