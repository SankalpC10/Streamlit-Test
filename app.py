# Importing necessary libraries
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reading in the movie data
movies_df = pd.read_csv('tmdb_5000_movies.csv')

# Creating a TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fitting and transforming the TF-IDF vectorizer on the 'overview' column
tfidf_matrix = tfidf.fit_transform(movies_df['overview'].fillna(''))

# Computing the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Creating a reverse mapping of movie titles and their indices
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

# Defining a function to get recommendations based on movie title
def get_recommendations(title):
    # Getting the index of the movie that matches the title
    idx = indices[title]

    # Getting the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sorting the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Getting the top 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Getting the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Returning the top 10 most similar movies
    return movies_df['title'].iloc[movie_indices].values

st.title('Movie Recommendor System')
#Testing the recommendation function
selected_movie_name =st.selectbox('How would you liked to be contacted?',movies_df['title'].values)
if st.button('Recommend'):
    recommendations =get_recommendations(selected_movie_name)
    st.text(recommendations[0])
    st.text(recommendations[1])
    st.text(recommendations[2])
    st.text(recommendations[3])
    st.text(recommendations[4])


# title = st.text_input('Movie title', 'Avatar')
# if st.button('Recommend'):
#     recommendations =get_recommendations(title)
#     st.text(recommendations[0])
#     st.text(recommendations[1])
#     st.text(recommendations[2])
#     st.text(recommendations[3])
#     st.text(recommendations[4])
#     st.text(recommendations[5])
#     st.text(recommendations[6])
#     st.text(recommendations[7])
#     st.text(recommendations[8])
#     st.text(recommendations[9])
