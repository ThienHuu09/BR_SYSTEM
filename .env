import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from PIL import Image
import openai
import os

# Load OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

@st.cache_data
def load_data():
    books_df = pd.read_csv('books.csv', usecols=[
        'title', 'description', 'authors', 'categories', 'published_year', 'ratings_count', 'thumbnail'
    ])
    return books_df

@st.cache_data
def compute_similarity(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    combined_features = (
        df['title'].fillna('') + ' ' +
        df['description'].fillna('') + ' ' +
        df['authors'].fillna('') + ' ' +
        df['categories'].fillna('')
    )
    tfidf_matrix = vectorizer.fit_transform(combined_features)
    sparse_tfidf = csr_matrix(tfidf_matrix)
    similarity_matrix = cosine_similarity(sparse_tfidf, sparse_tfidf)
    return similarity_matrix

def gpt4_search(query):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Find relevant book titles for the following search query: {query}",
        max_tokens=100
    )
    return response.choices[0].text.strip()

def get_recommendations(title, similarity_matrix, books_df, threshold=0.8, top_n=10):
    idx = books_df[books_df['title'].str.lower().str.contains(title.lower())].index
    if len(idx) == 0:
        gpt_suggestion = gpt4_search(title)
        return [(None, gpt_suggestion)]
    similarities = []
    for i in idx:
        book_similarities = list(enumerate(similarity_matrix[i]))
        similarities.extend(book_similarities)
    similarities = list(set(similarities))  # Remove duplicates
    filtered_books = [
        (i, score) for i, score in similarities if score >= threshold
    ]
    filtered_books = sorted(filtered_books, key=lambda x: x[1], reverse=True)[:top_n]
    recommendations = [
        (books_df.iloc[i], score) for i, score in filtered_books
    ]
    return recommendations

def display_cover(image_url):
    if pd.notna(image_url) and image_url.strip():
        return image_url  # Use the image URL directly
    else:
        return 'cover-not-found.jpg'

# Streamlit UI
st.title("📚 Book Recommendation System")
books_df = load_data()
similarity_matrix = compute_similarity(books_df)

# Display Top 50 Popular Books
st.subheader("🔥 Top 50 Most Popular Books:")
top_books = books_df.sort_values(by='ratings_count', ascending=False).head(50)
for i in range(0, 50, 5):
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        if i + idx < len(top_books):
            book = top_books.iloc[i + idx]
            with col.expander(f"{book['title']}"):
                st.image(display_cover(book.get('thumbnail', '')), width=100)
                st.write(f"**Author:** {book.get('authors', 'Unknown')}")
                st.write(f"**Published Year:** {book.get('published_year', 'Unknown')}")
                st.write(f"**Category:** {book.get('categories', 'Unknown')}")
                st.write(f"**Description:** {book.get('description', 'No description available')}")
                st.write(f"**Ratings Count:** {book.get('ratings_count', 0)}")

# User Input
user_input = st.text_input("🔍 Enter a keyword (title, description, category, published year)")
num_recommendations = st.number_input("Number of recommendations", min_value=1, max_value=50, value=10, step=1)

if st.button("Search"):
    recommendations = get_recommendations(user_input, similarity_matrix, books_df, threshold=0.8, top_n=num_recommendations)
    if recommendations:
        st.subheader(f"📖 Recommendations for '{user_input}':")
        for book, score in recommendations:
            if book is not None:
                with st.expander(f"{book['title']} (🔗 Similarity: {score:.2f})"):
                    st.image(display_cover(book.get('thumbnail', '')), width=100)
                    st.write(f"**Author:** {book.get('authors', 'Unknown')}")
                    st.write(f"**Published Year:** {book.get('published_year', 'Unknown')}")
                    st.write(f"**Category:** {book.get('categories', 'Unknown')}")
                    st.write(f"**Description:** {book.get('description', 'No description available')}")
                    st.write(f"**Ratings Count:** {book.get('ratings_count', 0)}")
            else:
                st.warning(f"🔍 {score}")
    else:
        st.warning("❌ No suitable recommendations found.")
