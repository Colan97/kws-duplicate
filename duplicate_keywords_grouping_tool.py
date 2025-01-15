import streamlit as st
import pandas as pd
import re
from difflib import SequenceMatcher
from collections import defaultdict
import nltk
from nltk.stem import PorterStemmer

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
    """Normalize keyword by removing common variations."""
    keyword = keyword.lower().strip()
    keyword = re.sub(r's\b', '', keyword)  # remove trailing 's'
    keyword = re.sub(r'\b(for|the|a|an)\b', '', keyword)  # remove common articles
    keyword = re.sub(r'\s+', ' ', keyword).strip()  # remove extra spaces
    return keyword


def process_keywords(keywords, similarity_threshold=0.85):
    """
    Group similar keywords together using a Jaccard-like approach (stems intersection over union).
    """
    keywords = [kw.strip() for kw in keywords if kw.strip()]
    groups = defaultdict(list)
    group_counter = 1
    stemmer = PorterStemmer()
    assigned_keywords = set()  # Keep track of assigned keywords

    for kw in keywords:
        if kw in assigned_keywords:
            continue

        normalized_kw = normalize_keyword(kw)
        stemmed_words = {stemmer.stem(word) for word in normalized_kw.split()}

        assigned = False
        for group_id, group_kws in groups.items():
            for group_kw in group_kws:
                normalized_group_kw = normalize_keyword(group_kw)
                stemmed_group_words = {stemmer.stem(word) for word in normalized_group_kw.split()}

                intersection = len(stemmed_words.intersection(stemmed_group_words))
                union = len(stemmed_words.union(stemmed_group_words))
                similarity = intersection / union if union else 0.0

                if similarity >= similarity_threshold:
                    groups[group_id].append(kw)
                    assigned_keywords.add(kw)
                    assigned = True
                    break
            if assigned:
                break

        if not assigned:
            groups[group_counter].append(kw)
            assigned_keywords.add(kw)
            group_counter += 1

    return groups


def main():
    st.title("Keyword Grouping Tool")
    st.write("This app groups similar keywords using an adjustable similarity threshold. "
             "Choose how to input keywords, set your threshold, and click **Group Keywords** to see results.")

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
            # Accept both commas and line breaks
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
            # Accept both commas and line breaks
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

        # Process the keywords
        groups = process_keywords(keywords, threshold)

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
