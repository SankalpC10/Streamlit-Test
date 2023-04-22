import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import ast
import warnings; warnings.simplefilter('ignore')

# def fetch_poster(movie_id):
#    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=72e60e8287ff77423bfa32b8c904e3dd&language=en-US'.format(movie_id))
#    data=response.json()
#    #print("https://api.themoviedb.org/3/movie/"+data['poster_path'])
#    return "https://api.themoviedb.org/3/movie/"+data['poster_path']

movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')
movies=movies.merge(credits,on='title')
movies=movies[['movie_id','title','genres','overview','keywords','cast','crew']]
movies.dropna(inplace=True)

def json_to_list(obj):
    values=[]
    for i in ast.literal_eval(obj):
        values.append(i['name'])
    return values

movies['genres']=movies.genres.apply(json_to_list)
movies['keywords']=movies.keywords.apply(json_to_list)

def top_cast(obj):
    values=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter<3:
            values.append(i['name'])
            counter+=1
        else:
            break
    return values

movies['cast']=movies.cast.apply(top_cast)

def director(obj):
    values=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            values.append(i['name'])
            break
    return values

movies['crew']=movies.crew.apply(director)

movies['overview']=movies.overview.apply(lambda x:x.split())

def concat_str(s):
    s=[i.replace(" ","") for i in s]
    return s

movies['genres']=movies.genres.apply(concat_str)
movies['keywords']=movies.keywords.apply(concat_str)
movies['cast']=movies.cast.apply(concat_str)
movies['crew']=movies.crew.apply(concat_str)

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

df=movies[["movie_id","title","tags"]]

df['tags']=df['tags'].apply(lambda x:" ".join(x))

cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(df['tags']).toarray()

ps=PorterStemmer()
def stem(s):
    v=[]
    for i in s.split():
        v.append(ps.stem(i))
    return " ".join(v)

df['tags']=df['tags'].apply(stem)

similarity_matrix=cosine_similarity(vectors)

st.title('Movie Recommendor System')
def recommend(movie):
    movie_index = df[df['title'] == movie].index[0]
    distances = similarity_matrix[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        st.text(df.iloc[i[0]].title)
# def recommend_movie(movie):
#     movie_index = movies[movies['title'] == movie].index[0]
#     distances = similarity_matrix[movie_index]
#     movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
#     recommended_movies=[]
#     recommended_movies_posters=[]
#     for i in movies_list:
#         movie_id=movies.iloc[i[0]].movie_id
#         recommended_movies.append(movies.iloc[i[0]].title)
#         # recommended_movies_posters.append(fetch_poster(movie_id))
#     return recommended_movies


#movies_dict = pickle.load(open('movie_dict.pkl','rb'))
# md=open('movie_dict.pkl','rb')
# movies_dict=pickle.load(md)
# movies = pd.DataFrame(movies_dict)
# similarity_matrix=pickle.load(open('similarity_matrix.pkl','rb'))



selected_movie_name =st.selectbox('How would you liked to be contacted?',movies['title'].values)
if st.button('Recommend'):
    recommend(selected_movie_name)
#     recommendations =recommend(selected_movie_name)
#     st.text(recommendations[0])
#     st.text(recommendations[1])
#     st.text(recommendations[2])
#     st.text(recommendations[3])
#     st.text(recommendations[4])