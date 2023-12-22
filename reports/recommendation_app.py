import streamlit as st
import pandas as pd
import joblib
import pickle
from gensim.models import Word2Vec


# Load your dataset
@st.cache_data   #st.cache_resource #
def load_data():
    return pickle.load(open('model/df_recommendation_similarity.pkl', 'rb'))

df = load_data()


# Load your model
@st.cache_data #(allow_output_mutation=True)
def load_model():
    return joblib.load('model/model_similarity.joblib')


model = load_model()

# Define the function that will retrieve similar words
def rec(target_word):
    similar_words = model.wv.most_similar(target_word, topn=5)
    return similar_words

# Function to get recommendations and other data
def get_recommendations(food_item):
    # Use your loaded model to find similar words
    # Assuming the model has a method 'most_similar' that takes a word and returns similar words
    similar_words = model.rec(food_item)
    return similar_words


def get_avg_stars(food_item):
    # Replace this logic with the actual calculation from your dataset
    return df[df['food mentioned'] == food_item]['stars'].mean()


def get_total_reviews(food_item):
    # Replace this logic with the actual calculation from your dataset
    return df[df['food mentioned'] == food_item]['id_review'].count()

# Function to get sentiment for a word
def get_sentiment(word):
    return df.loc[df['food mentioned'] == word]['sentiment'].iloc[0]



### Streamlit layout
st.title('🥐 Menu Recommendation System')

# Dropdown for food selection
food_options = df['food mentioned'].unique()  # Replace 'food_column' with your actual column name for food items
selected_food = st.selectbox('Type or select a food word from the dropdown', food_options)

if st.button('Show the recommendation'):
    # Get similar words, reviews, average stars, and sentiment for the selected food
    similar_words = rec(selected_food)
    avg_stars = get_avg_stars(selected_food)
    total_reviews = get_total_reviews(selected_food)
    sentiment = get_sentiment(selected_food)
    #sentiment = get_sentiment(similar_words)

    # Display the similar words
    st.subheader('Similar Food Recommendations:')
    for word, similarity in similar_words:
        st.write(f"{word} (Similarity: {similarity:.2f})")


    # Display other information
    st.subheader('Food Details:')
    st.write(f"Avg Stars: {avg_stars:.2f}")
    st.write(f"Total of Reviews: {total_reviews}")
    st.write(f"Overall Experience: {sentiment}")

### Background
import base64
import os


# Function to get image base64
def get_image_base64(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Function to set a background image in the sidebar
def set_sidebar_background(image_path):
    # Get base64 encoding of the image
    image_base64 = get_image_base64(image_path)

    # Set the background using the base64 string
    sidebar_style = f"""
        <style>
        [data-testid="stSidebar"] > div:first-child {{
            background-image: url("data:image/png;base64,{image_base64}");
            background-size: cover;
        }}
        </style>
    """
    st.markdown(sidebar_style, unsafe_allow_html=True)


# Set the image in the sidebar
image_path = "imgs/img1.png"  # Adjust the path to where the image is stored
set_sidebar_background(image_path)

# Your Streamlit app code goes here
st.sidebar.title("")
st.sidebar.write("")

#streamlit run reports/recommendation_app.py
