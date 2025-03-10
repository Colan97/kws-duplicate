import streamlit as st
import pandas as pd
import re
from collections import defaultdict
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# For the cosine similarity method, scikit-learn is required.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    st.error("scikit-learn is required for the cosine similarity method. Please install it via pip (pip install scikit-learn).")

# Ensure nltk data is available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


def normalize_keyword(keyword):
    """
    Normalize a keyword by converting to lowercase, removing punctuation,
    common articles, extra spaces, and then applying lemmatization.
    """
    keyword = keyword.lower().strip()
    # Remove punctuation
    keyword = re.sub(r'[^\w\s]', '', keyword)
    # Remove common articles
    keyword = re.sub(r'\b(for|the|a|an)\b', '', keyword)
    # Remove extra spaces
    keyword = re.sub(r'\s+', ' ', keyword).strip()
    # Lemmatize each word
    lemmatizer = WordNetLemmatizer()
    normalized_words = [lemmatizer.lemmatize(word) for word in keyword.split()]
    return ' '.join(normalized_words)


def process_keywords(keywords, similarity_threshold=0.85, method='stem'):
    """
    Group similar keywords together using one of two methods:
    - 'stem': A basic stem-based Jaccard-like similarity.
    - 'cosine': Cosine similarity on TF-IDF vectors.
    """
    keywords = [kw.strip() for kw in keywords if kw.strip()]
    groups = defaultdict(list)
    group_counter = 1
    assigned = [False] * len(keywords)
    
    if method == 'stem':
        stemmer = PorterStemmer()
        for i, kw in enumerate(keywords):
            if assigned[i]:
                continue
            normalized_kw = normalize_keyword(kw)
            stemmed_words = {stemmer.stem(word) for word in normalized_kw.split()}
            groups[group_counter].append(kw)
            assigned[i] = True
            for j in range(i + 1, len(keywords)):
                if not assigned[j]:
                    normalized_kw2 = normalize_keyword(keywords[j])
                    stemmed_words2 = {stemmer.stem(word) for word in normalized_kw2.split()}
                    intersection = len(stemmed_words.intersection(stemmed_words2))
                    union = len(stemmed_words.union(stemmed_words2))
                    similarity = intersection / union if union else 0.0
                    if similarity >= similarity_threshold:
                        groups[group_counter].append(keywords[j])
                        assigned[j] = True
            group_counter += 1

    elif method == 'cosine':
        # Create a list of normalized keywords for vectorization
        normalized_keywords = [normalize_keyword(kw) for kw in keywords]
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(normalized_keywords)
        sim_matrix = cosine_similarity(tfidf)
        for i in range(len(keywords)):
            if assigned[i]:
                continue
            groups[group_counter].append(keywords[i])
            assigned[i] = True
            for j in range(i + 1, len(keywords)):
                if not assigned[j]:
                    if sim_matrix[i, j] >= similarity_threshold:
                        groups[group_counter].append(keywords[j])
                        assigned[j] = True
            group_counter += 1

    else:
        st.error("Invalid similarity method selected. Please choose 'stem' or 'cosine'.")
        
    return groups


def main():
    st.title("Enhanced Keyword Grouping Tool")
    st.write(
        "This app groups similar keywords using an adjustable similarity threshold and two different "
        "similarity methods. Choose your input method, select a similarity method, adjust your threshold, "
        "and click **Group Keywords** to see results."
    )

    # --- Select input method ---
    input_method = st.selectbox(
        "How would you like to input keywords?",
        ("Paste a list of keywords", "Upload a file", "Enter keywords one by one")
    )

    keywords = []

    # --- 1) Paste a list of keywords ---
    if input_method == "Paste a list of keywords":
        st.write("Paste or type your keywords below (comma-separated or line-separated).")
        text_input = st.text_area("Keywords")
        if text_input:
            raw_list = []
            for line in text_input.splitlines():
                raw_list.extend(line.split(","))
            keywords = [k.strip() for k in raw_list if k.strip()]

    # --- 2) Upload a file ---
    elif input_method == "Upload a file":
        st.write("Upload a text file (one keyword per line, or comma-separated).")
        uploaded_file = st.file_uploader("Upload your file", type=["txt", "csv", "tsv"])
        if uploaded_file is not None:
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            raw_list = []
            for line in content.splitlines():
                raw_list.extend(line.split(","))
            keywords = [k.strip() for k in raw_list if k.strip()]

    # --- 3) Enter keywords one by one ---
    else:
        st.write("Type each keyword on a new line below.")
        text_input = st.text_area("Keywords (one per line)")
        if text_input:
            keywords = [line.strip() for line in text_input.splitlines() if line.strip()]

    # --- Select similarity method ---
    similarity_method = st.selectbox(
        "Select similarity method:",
        ("Basic Stem-based", "Cosine Similarity (TF-IDF)")
    )
    # Map selection to internal method identifier
    method_identifier = "stem" if similarity_method == "Basic Stem-based" else "cosine"

    # --- Slider for similarity threshold ---
    threshold = st.slider(
        "Set the similarity threshold (0.70 to 0.95)",
        min_value=0.70,
        max_value=0.95,
        value=0.85,
        step=0.01
    )

    # --- Button to trigger grouping ---
    if st.button("Group Keywords"):
        if not keywords:
            st.warning("No keywords provided. Please add some keywords first.")
            return

        # Process the keywords using the selected similarity method
        groups = process_keywords(keywords, similarity_threshold=threshold, method=method_identifier)

        # Create a DataFrame of group -> keyword
        rows = []
        for group_id, group_keywords in groups.items():
            for kw in group_keywords:
                rows.append({'Group': group_id, 'Keyword': kw})
        df = pd.DataFrame(rows)
        df = df.sort_values(['Group', 'Keyword']).reset_index(drop=True)

        # --- Display the results ---
        st.subheader("Grouped Keywords")
        st.dataframe(df, use_container_width=True)

        # --- Provide a download button for CSV ---
        st.subheader("Download Results")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="keyword_groups.csv",
            mime="text/csv"
        )


# Run the Streamlit app
if __name__ == "__main__":
    main()
