import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Page Configuration and Title ---

st.set_page_config(
    page_title="Library Book Recommender",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Library Book Recommendation System")
st.markdown("---")

# --- 2. File Uploader and Data Processing ---

st.header("Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Please upload a CSV file with UserID, BookTitle, Genre, Author, and PublicationYear columns.")

# Initialize session state for caching
if 'df' not in st.session_state:
    st.session_state.df = None
if 'cosine_sim' not in st.session_state:
    st.session_state.cosine_sim = None
if 'book_indices' not in st.session_state:
    st.session_state.book_indices = None

@st.cache_data
def load_and_process_data(file_object):
    """
    Loads the uploaded CSV dataset, cleans it, and performs all pre-computation.
    Includes a fix to reset the index after dropping rows.
    """
    with st.spinner("Loading and processing data..."):
        try:
            df = pd.read_csv(file_object)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None, None, None

        st.success("Dataset loaded successfully!")

        # Data Cleaning and Transformation
        
        df.columns = [col.strip() for col in df.columns]
        df['Author'].fillna('Unknown', inplace=True)
        df['Genre'] = df['Genre'].str.lower().fillna('unknown')
        
        # Ensure PublicationYear is a number
        df['PublicationYear'] = pd.to_numeric(df['PublicationYear'], errors='coerce')
        
        # Drop rows with any other missing values
        df.dropna(subset=['BookTitle', 'UserID', 'PublicationYear'], inplace=True)
        
        # --- CRITICAL FIX: Reset the index after dropping rows ---
        df.reset_index(drop=True, inplace=True)

        if df.empty:
            st.error("The dataset is empty after cleaning. Cannot create recommendations.")
            return None, None, None

        # Create a combined feature column for content-based filtering
        df['Features'] = df['Genre'] + ' ' + df['Author']
        
        # Vectorize the 'Features' column
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['Features'])

        # Calculate content similarity
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Create a mapping from book title to index
        book_indices = pd.Series(df.index, index=df['BookTitle'])

    return df, cosine_sim, book_indices

if uploaded_file:
    # Process the data only if it hasn't been done yet in this session
    if st.session_state.df is None:
        st.session_state.df, st.session_state.cosine_sim, st.session_state.book_indices = load_and_process_data(uploaded_file)
    
    # Check if data processing was successful
    if st.session_state.df is not None:
        # --- 3. Sidebar for User Selection and Filtering ---
        with st.sidebar:
            st.header("Filters and User Selection")

            # Genre filter
            all_genres = sorted(st.session_state.df['Genre'].unique())
            selected_genres = st.multiselect(
                "Filter by Genre(s)",
                options=all_genres,
                default=[]
            )

            # Publication Year filter
            min_year = int(st.session_state.df['PublicationYear'].min())
            max_year = int(st.session_state.df['PublicationYear'].max())
            selected_year_range = st.slider(
                "Filter by Publication Year",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )

            # User Selection
            st.markdown("---")
            user_ids = sorted(st.session_state.df['UserID'].unique())
            selected_user_id = st.selectbox(
                'Select a User ID to get recommendations:',
                user_ids,
                key='user_selector'
            )
            get_rec_button = st.button("Get Recommendations", use_container_width=True)

        # --- Recommendation Model Logic ---
        
        def get_recommendations(user_id, df, cosine_sim, book_indices, num_recommendations=5):
            """
            Generates book recommendations for a given user based on content-based filtering.
            """
            user_books = df[df['UserID'] == user_id]['BookTitle'].tolist()
            if not user_books:
                return []

            recommendations = {}
            for book in user_books:
                try:
                    # Get the first index for a book, even if it has duplicates
                    idx = book_indices[book].iloc[0] if isinstance(book_indices[book], pd.Series) else book_indices[book]
                except KeyError:
                    continue

                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:11]
                
                book_indices_to_recommend = [i[0] for i in sim_scores]
                for i in book_indices_to_recommend:
                    recommended_book_title = df.iloc[i]['BookTitle']
                    if recommended_book_title not in user_books:
                        if recommended_book_title not in recommendations:
                            recommendations[recommended_book_title] = 0
                        recommendations[recommended_book_title] += df.iloc[i]['BorrowedBooks'] * sim_scores[book_indices_to_recommend.index(i)][1]

            sorted_recommendations = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)
            recommended_book_titles = [book for book, score in sorted_recommendations]

            return recommended_book_titles[:num_recommendations]

        # --- 4. Main Content Area: Visualization and Recommendations ---
        col1, col2 = st.columns(2)

        with col1:
            st.header("Top Borrowed Books")
            book_borrow_counts = st.session_state.df['BookTitle'].value_counts()
            top_n = 5
            top_books = book_borrow_counts.head(top_n)

            fig, ax = plt.subplots(figsize=(10, 6))
            top_books.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(f'Top {top_n} Most Borrowed Books', fontsize=16)
            ax.set_xlabel('Book Title', fontsize=12)
            ax.set_ylabel('Number of Borrows', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            if get_rec_button:
                # Filter the main DataFrame based on user's selections
                filtered_df = st.session_state.df.copy()
                if selected_genres:
                    filtered_df = filtered_df[filtered_df['Genre'].isin(selected_genres)]
                
                filtered_df = filtered_df[
                    (filtered_df['PublicationYear'] >= selected_year_range[0]) & 
                    (filtered_df['PublicationYear'] <= selected_year_range[1])
                ]
                
                if filtered_df.empty:
                    st.error("No books match your selected filters. Please adjust your criteria.")
                else:
                    # Re-create the TF-IDF matrix and cosine similarity for the filtered data
                    tfidf = TfidfVectorizer(stop_words='english')
                    filtered_tfidf_matrix = tfidf.fit_transform(filtered_df['Features'])
                    filtered_cosine_sim = linear_kernel(filtered_tfidf_matrix, filtered_tfidf_matrix)
                    filtered_book_indices = pd.Series(filtered_df.index, index=filtered_df['BookTitle'])

                    st.header(f'Recommendations for User {selected_user_id}')
                    with st.spinner('Generating recommendations...'):
                        recommended_books = get_recommendations(
                            selected_user_id,
                            filtered_df,
                            filtered_cosine_sim,
                            filtered_book_indices
                        )

                    if recommended_books:
                        book_details = filtered_df[filtered_df['BookTitle'].isin(recommended_books)].drop_duplicates(subset='BookTitle').reset_index(drop=True)
                        
                        st.markdown("Here are some books we think you'll love based on your past borrowings and selected filters:")
                        for index, row in book_details.iterrows():
                            st.markdown(f"**- {row['BookTitle']}** by {row['Author']} (Genre: {row['Genre']}) (Year: {row['PublicationYear']})")
                    else:
                        st.info("No recommendations found for this user with the current filters.")

else:
    st.info("Please upload a CSV file to get started.")


