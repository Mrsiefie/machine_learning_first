from flask import Flask, render_template, request, jsonify

import pickle
import pandas as pd

app = Flask(__name__)



movies_list = pickle.load(open('movies.pkl','rb'))
movies_list =movies_list['title'].values

similarity = pickle.load(open('similarity.pkl','rb'))
def recommendss(selectedValue):
    Movie_index= movies_list[movies_list['title']=='selectedValue'].index[0]
    distances =similarity[Movie_index]
    movies_list= sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    recommended_movie = []
    for i in movies_list:
        recommended_movie.append(movies_list.iloc[i[0]].title)
    return  recommended_movie


@app.route('/')
def hello_world():  # put application's code here
    options = movies_list.tolist()
    return render_template('index.html', options=options)



@app.route('/recommend', methods=['POST'])
def recommend():
    selected_movie = request.form.get('movie')


    # Process the selected movie using your machine learning model
    # and get movie recommendations
    recommendations = selected_movie

    return render_template('index.html', recommendations=recommendations)



if __name__ == '__main__':
    app.run(debug=True)
