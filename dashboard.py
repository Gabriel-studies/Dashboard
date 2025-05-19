import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns




# Page layout
st.set_page_config(page_title="Simple Movie Recommender", layout="wide")

# Load Data
movies = pd.read_csv('movies.csv', encoding='latin1')
ratings = pd.read_csv('rating.csv', encoding='latin1')

# Drop NaNs
movies.dropna(inplace=True)
ratings.dropna(inplace=True)

# Pivot Table
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Euclidean Similarity
euc_dist = euclidean_distances(user_movie_matrix)
np.fill_diagonal(euc_dist, np.inf)
euc_sim = 1 / (1 + euc_dist)
euc_sim_df = pd.DataFrame(euc_sim, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Recommendation function
def recommend_movies(user_id, top_n=5):
    similar_users = euc_sim_df[user_id].sort_values(ascending=False).head(5).index
    top_users_ratings = user_movie_matrix.loc[similar_users]
    average_ratings = top_users_ratings.mean()
    user_seen_movies = user_movie_matrix.loc[user_id]
    unrated_movies = average_ratings[user_seen_movies == 0]
    return unrated_movies.sort_values(ascending=False).head(top_n)

# Sidebar Login
user_id = st.sidebar.selectbox(
    'Login:', 
    options=[None] + list(ratings['userId'].unique()), 
    index=0
)

if user_id is None:
    st.sidebar.warning("Please select a user to log in.")
    st.stop()

# Recommendations
st.title(f"Hi {user_id}, here are your movie picks 🎬")

top_recommendations = recommend_movies(user_id)
recommended_movies = movies[movies['movieId'].isin(top_recommendations.index)]
top_movies = top_recommendations.reset_index()
top_movies.columns = ['movieId', 'Predicted Rating']
top_movies = top_movies.merge(movies[['movieId', 'title', 'genres']], on='movieId')
top_movies['genre_list'] = top_movies['genres'].str.split('|')
top_movies_exploded = top_movies.explode('genre_list')
available_genres = sorted(top_movies_exploded['genre_list'].unique())

# Genre Filter
selected_genre = st.selectbox('Filter by Genre:', options=['All'] + available_genres)

filtered_movies = (
    top_movies_exploded[top_movies_exploded['genre_list'] == selected_genre]
    if selected_genre != 'All'
    else top_movies_exploded
)

filtered_movies = filtered_movies.drop_duplicates(subset='movieId').sort_values(by='Predicted Rating', ascending=False)
filtered_movies['title_with_genre'] = filtered_movies['title'] + ' [' + filtered_movies['genre_list'] + ']'

# Recommendation Chart
fig_bar = px.bar(
    filtered_movies,
    x='Predicted Rating',
    y='title_with_genre',
    orientation='h',
    text='title_with_genre'
)
fig_bar.update_traces(textposition='inside', marker_color='steelblue')
fig_bar.update_layout(
    yaxis=dict(autorange="reversed"),
    showlegend=False,
    xaxis_visible=False,
    yaxis_visible=False,
    title='🎯 Recommended Movies',
    margin=dict(l=20, r=20, t=30, b=20),
    height=400
)
st.plotly_chart(fig_bar, use_container_width=True)

# User Preferences
st.markdown("### 🎥 Your Movie Preferences Analytics")

col1, col2 = st.columns([1.5, 1])
with col1:
    user_ratings = ratings[ratings['userId'] == user_id]
    user_ratings_data = user_ratings['rating'].value_counts().sort_index().reset_index()
    user_ratings_data.columns = ['Rating', 'Count']
    fig_hist = px.bar(user_ratings_data, x='Rating', y='Count', title='Your Rating Distribution', text='Count')
    fig_hist.update_traces(marker_color='steelblue', textposition='outside')
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    genre_counts = filtered_movies['genre_list'].value_counts().head(5).reset_index()
    genre_counts.columns = ['Genre', 'Count']
    fig_pie = px.pie(genre_counts, values='Count', names='Genre', title='Top Genres')
    st.plotly_chart(fig_pie, use_container_width=True)



# --------------------------------- Heatmap of similar users ------------------------ #

if st.checkbox("👥 See Similar Users Heatmap"):
    st.markdown("#### Heatmap of Your Top 5 Similar Users")

    # Top 5 similar users
    sim_users = euc_sim_df.loc[user_id].sort_values(ascending=False).head(5).index
    sim_matrix = euc_sim_df.loc[sim_users, sim_users]

    # Custom blues scales including our 'steelblue'
    custom_blues = LinearSegmentedColormap.from_list(
        'custom_steelblue', 
        ['#e0f7fa', 'steelblue', '#08306b']
    )

    # Display
    fig, ax = plt.subplots()
    ax.set_title("Top 5 Similar Users Heatmap (Darker = More Similar)")
    sns.heatmap(sim_matrix, annot=True, cmap=custom_blues, ax=ax)
    st.pyplot(fig)
