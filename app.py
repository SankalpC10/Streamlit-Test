import pickle
import pandas as pd
import streamlit as st
import requests


# def fetch_poster(movie_id):
#    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=72e60e8287ff77423bfa32b8c904e3dd&language=en-US'.format(movie_id))
#    data=response.json()
#    #print("https://api.themoviedb.org/3/movie/"+data['poster_path'])
#    return "https://api.themoviedb.org/3/movie/"+data['poster_path']


def recommend_movie(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity_matrix[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies=[]
    recommended_movies_posters=[]
    for i in movies_list:
        movie_id=movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        # recommended_movies_posters.append(fetch_poster(movie_id))
    return recommended_movies

#movies_dict = pickle.load(open('movie_dict.pkl','rb'))
md=open('movie_dict.pkl','rb')
movies_dict=pickle.load(md)
movies = pd.DataFrame(movies_dict)
similarity_matrix=pickle.load(open('similarity_matrix.pkl','rb'))


st.title('Movie Recommendor System')

selected_movie_name =st.selectbox('How would you liked to be contacted?',movies['title'].values)

if st.button('Recommend'):
    recommendations =recommend_movie(selected_movie_name)
    st.text(recommendations[0])
    st.text(recommendations[1])
    st.text(recommendations[2])
    st.text(recommendations[3])
    st.text(recommendations[4])