import streamlit as st
import pandas as pd
import logging
import numpy as np
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, NUMERIC
from whoosh.qparser import MultifieldParser
from diskcache import Cache
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import ollama
import os
from pathlib import Path
import time
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cấu hình Streamlit
st.set_page_config(layout="wide", page_title="Book Recommendation ", initial_sidebar_state="expanded")

# Đường dẫn index Whoosh và cache
INDEX_DIR = Path("indexdir")
CACHE_DIR = Path("./cache")

# Tạo schema Whoosh
schema = Schema(
    title=TEXT(stored=True),
    authors=TEXT(stored=True),
    published_year=NUMERIC(stored=True),
    category=TEXT(stored=True),
    description=TEXT(stored=True),
    thumbnail=TEXT(stored=True),
    ratings_count=NUMERIC(stored=True)
)

# Khởi tạo index Whoosh
def create_or_open_index(books_df):
    if not INDEX_DIR.exists():
        os.makedirs(INDEX_DIR)
        ix = create_in(INDEX_DIR, schema)
        writer = ix.writer()
        for _, row in books_df.iterrows():
            writer.add_document(
                title=str(row['title']),
                authors=str(row['authors']),
                published_year=int(row['published_year']) if pd.notna(row['published_year']) else 0,
                category=str(row['categories']),
                description=str(row['description']),
                thumbnail=str(row['thumbnail'] if pd.notna(row['thumbnail']) else ''),
                ratings_count=float(row['ratings_count']) if pd.notna(row['ratings_count']) else 0
            )
        writer.commit()
        logger.info(f"Created new Whoosh index in {INDEX_DIR}")
    else:
        ix = open_dir(INDEX_DIR)
    return ix

# Khởi tạo cache
cache = Cache(CACHE_DIR)

# Số luồng tìm kiếm song song
executor = ThreadPoolExecutor(max_workers=2)

# Cache dữ liệu từ books.csv
@st.cache_data(hash_funcs={pd.DataFrame: lambda x: hash(x.to_string())})
def load_data():
    try:
        if not os.path.exists('books.csv'):
            logger.error("File 'books.csv' not found")
            st.error("Không tìm thấy file 'books.csv'. Vui lòng kiểm tra lại.")
            return pd.DataFrame()
        books_df = pd.read_csv('books.csv', usecols=[
            'title', 'description', 'authors', 'categories',
            'published_year', 'ratings_count', 'thumbnail'
        ], dtype={
            'title': 'string', 'description': 'string', 'authors': 'string',
            'categories': 'string', 'published_year': 'Int32',
            'ratings_count': 'float32', 'thumbnail': 'string'
        })
        books_df.fillna({
            'description': 'None', 'authors': 'None',
            'categories': 'None', 'thumbnail': ''
        }, inplace=True)
        books_df['ratings_count'] = books_df['ratings_count'].fillna(0)
        logger.info(f"Loaded dataset with {len(books_df)} books")
        return books_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error("Không thể tải dữ liệu sách")
        return pd.DataFrame()

@st.cache_resource
def compute_similarity(df):
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000,
                                   max_df=1.0, min_df=1)  # Điều chỉnh max_df và min_df
        combined_features = df['title'] + ' ' + df['description']
        tfidf_matrix = vectorizer.fit_transform(combined_features)
        sparse_tfidf = csr_matrix(tfidf_matrix)
        similarity_matrix = cosine_similarity(sparse_tfidf)
        return similarity_matrix
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return None

@st.cache_data
def mistral_search(query):
    try:
        future = executor.submit(ollama.chat, model="mistral", messages=[
            {"role": "system", "content": "You are a book recommendation assistant. Only return titles that are highly relevant to the query based on scientific or factual content."},
            {"role": "user", "content": f"Find relevant book titles for: {query}"}
        ])
        response = future.result(timeout=10)
        return response["message"]["content"].strip()
    except TimeoutError:
        logger.warning(f"Mistral query for '{query}' timed out after 10s")
        return "Không thể tìm kiếm bằng Mistral"
    except Exception as e:
        logger.error(f"Mistral search error: {e}")
        return "Không thể tìm kiếm bằng Mistral"

def get_recommendations(query, similarity_matrix, books_df, ix):
    try:
        # Tìm kiếm với Whoosh trên nhiều trường
        with ix.searcher() as searcher:
            parser = MultifieldParser(["title", "description", "category"], schema=schema)  # Thêm category
            q = parser.parse(query)
            results = searcher.search(q, limit=5)  # Tăng limit để lấy nhiều kết quả hơn
            logger.debug(f"Whoosh results for '{query}': {len(results)} matches")
            if results:
                whoosh_results = []
                for r in results:
                    match = books_df[books_df['title'].str.lower() == r['title'].lower()]
                    book_data = match.iloc[0].to_dict() if not match.empty else {
                        'title': r['title'],
                        'description': r['description'],
                        'authors': r.get('authors', 'None'),
                        'published_year': r.get('published_year', 0),
                        'category': r.get('category', 'None'),
                        'thumbnail': r.get('thumbnail', ''),
                        'ratings_count': r.get('ratings_count', 0)
                    }
                    # Kiểm tra nếu từ khóa xuất hiện trong description hoặc category
                    if query.lower() in (book_data['description'].lower() if book_data['description'] else '') or \
                       query.lower() in (book_data['category'].lower() if book_data['category'] else ''):
                        whoosh_results.append(book_data)
                if whoosh_results:
                    return pd.DataFrame(whoosh_results)

        # Nếu không tìm thấy, dùng TF-IDF với kiểm tra liên quan
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, max_df=1.0, min_df=1)
        combined_features = books_df['title'] + ' ' + books_df['description'] + ' ' + books_df['category']
        tfidf_matrix_query = vectorizer.fit_transform([query])  # Vectorize query
        tfidf_matrix_books = vectorizer.transform(combined_features)
        logger.debug(f"TF-IDF matrix shape: {tfidf_matrix_books.shape}")
        similarity_scores = cosine_similarity(tfidf_matrix_query, tfidf_matrix_books).flatten()
        logger.debug(f"Similarity scores shape: {similarity_scores.shape}")

        # Lọc các sách có độ tương đồng cơ bản (>= 0.3) để đảm bảo liên quan
        relevant_indices = np.where(similarity_scores >= 0.1)[0]
        if len(relevant_indices) == 0:
            logger.debug(f"No books with basic relevance (>= 0.1) for '{query}'")
            return pd.DataFrame()

        # Lấy chỉ số có độ tương đồng >= 90% từ các sách liên quan
        high_similarity_indices = np.intersect1d(relevant_indices, np.where(similarity_scores >= 0.9)[0])
        logger.debug(f"TF-IDF high similarity scores for '{query}': {len(high_similarity_indices)} matches above 0.9")
        if len(high_similarity_indices) > 0:
            top_indices = high_similarity_indices[np.argsort(similarity_scores[high_similarity_indices])[-5:]]  # Top 5 cao nhất
            return books_df.iloc[top_indices]
        else:
            logger.debug(f"No books with high similarity (>= 0.9) for '{query}'")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return pd.DataFrame()

def display_cover(image_url):
    # Sử dụng chính xác mã bạn cung cấp, không thêm kiểm tra lỗi
    return image_url if pd.notna(image_url) and image_url.strip() else 'cover-not-found.jpg'

# Khởi tạo state (chỉ gọi một lần khi ứng dụng khởi động)
def init_state():
    if 'initialized' not in st.session_state:
        st.session_state.selected_book = None
        st.session_state.search_results = None  # Thêm biến để lưu kết quả tìm kiếm
        st.session_state.initialized = True
    logger.debug("Initialized session state: selected_book={}, search_results={}".format(
        st.session_state.selected_book, st.session_state.search_results))

# Hiển thị danh sách sách (top hoặc kết quả tìm kiếm)
def display_books(books_df, title="Danh sách sách"):
    st.subheader(title)
    logger.debug(f"Displaying {len(books_df)} books for {title} with columns: {books_df.columns.tolist()}")
    for i in range(0, min(len(books_df), 50), 5):  # Hiển thị tối đa 50 sách
        cols = st.columns(5)
        for j, col in enumerate(cols):
            if i + j < len(books_df):
                book = books_df.iloc[i + j].to_dict()
                with col:
                    st.image(display_cover(book['thumbnail']), width=150)
                    st.write(f"**{book['title']}**")
                    st.write(f"Rating: {int(book['ratings_count']) if pd.notna(book['ratings_count']) else 0}")
                    if st.button("Xem", key=f"{title.replace(' ', '_')}_{i+j}"):
                        logger.debug(f"View clicked for book: {book['title']} from {title}, data: {book}")
                        st.session_state.selected_book = book
                        st.rerun()

# Hiển thị thông tin chi tiết (pop-up)
def display_book_details(books_df, similarity_matrix):
    if st.session_state.selected_book is not None:
        book = st.session_state.selected_book
        logger.debug(f"Displaying details for book: {book['title']}, data: {book}")
        if not all(key in book for key in ['title', 'thumbnail', 'description', 'authors', 'published_year', 'categories', 'ratings_count']):
            st.error(f"Dữ liệu sách không đầy đủ: {book}")
            st.session_state.selected_book = None
            st.rerun()
            return
        st.markdown("---")
        st.subheader("📖 Chi tiết sách")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(display_cover(book['thumbnail']), width=300)
        with col2:
            st.write(f"**Tựa đề:** {book['title']}")
            st.write(f"**Tác giả:** {book.get('authors', 'Unknown')}")
            st.write(f"**Năm xuất bản:** {book.get('published_year', 'N/A')}")
            st.write(f"**Thể loại:** {book.get('categories', 'N/A')}")
            st.write(f"**Mô tả:** {book.get('description', 'No description available')}")
            st.write(f"**Số lượt đánh giá:** {int(book.get('ratings_count', 0))}")
        st.subheader("📚 Sách tương tự")
        try:
            idx = books_df[books_df['title'] == book['title']].index[0]
            similar_indices = similarity_matrix[idx].argsort()[::-1][1:6]  # 5 sách tương tự
            similar_books = books_df.iloc[similar_indices]
            for _, similar_book in similar_books.iterrows():
                st.write(f"- {similar_book['title']}")
        except Exception as e:
            logger.error(f"Error getting similar books: {e}")
            st.write("Không thể tải sách tương tự.")
        if st.button("🔙 Đóng"):
            logger.debug(f"Closing details for book: {book['title']}")
            st.session_state.selected_book = None
            st.rerun()

# Main UI
def main():
    # Chỉ gọi init_state một lần khi ứng dụng khởi động
    if 'initialized' not in st.session_state:
        init_state()

    st.title("📚 Book Recommendation System")

    # Load dữ liệu
    books_df = load_data()
    if books_df.empty:
        return
    similarity_matrix = compute_similarity(books_df)
    global ix
    try:
        ix = create_or_open_index(books_df)
    except Exception as e:
        logger.error(f"Error creating or opening index: {e}")
        st.error(f"Lỗi khi tạo hoặc mở index: {str(e)}")
        return

    # Ưu tiên hiển thị chi tiết nếu selected_book tồn tại
    if st.session_state.selected_book is not None:
        logger.debug("Showing details due to selected_book")
        display_book_details(books_df, similarity_matrix)
        return

    # Search UI
    with st.form(key='search_form', clear_on_submit=False):
        search_input = st.text_input("🔍 Nhập từ khóa sách")
        search_button = st.form_submit_button("Tìm kiếm")

    # Xử lý tìm kiếm
    if search_button and search_input:
        recommendations = get_recommendations(search_input, similarity_matrix, books_df, ix)
        logger.debug(f"Recommendations for '{search_input}': {len(recommendations)} books, columns: {recommendations.columns.tolist()}")
        if not recommendations.empty:
            st.session_state.search_results = recommendations  # Lưu kết quả tìm kiếm
            display_books(recommendations, f"📖 Kết quả tìm kiếm cho nội dung bạn tìm kiếm")
        else:
            st.write(f"Không tìm thấy sách nào liên quan đến nội dung bạn tìm kiếm .")

    # Hiển thị top sách hoặc kết quả tìm kiếm trước đó khi không ở chế độ chi tiết
    if st.session_state.selected_book is None:
        if st.session_state.search_results is not None:
            display_books(st.session_state.search_results, f"📖 Kết quả tìm kiếm gần đây")
        else:
            display_books(books_df.nlargest(50, 'ratings_count'), "🔥 Top 50 sách nổi bật")

if __name__ == "__main__":
    main()