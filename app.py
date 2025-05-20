import streamlit as st
import pandas as pd
from recommender import get_recommendations, prepare_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df, cosine_sim = prepare_data()
 

# App UI

st.set_page_config(page_title = "Netflix Recommender", page_icon=":movie_camera:", layout="wide")

st.title("🎬 Netflix Movie/TV Show Recommendation Engine")

content_type = st.selectbox("Select content Type", ["Movie", "TV Show"])

selected_title = st.text_input(f"Enter a {content_type} title:")
df_filtered = df[df["type"] == content_type].reset_index(drop=True)

if selected_title:
    # Recompute cosine similarity matrix on the filtered df ONLY
    df_filtered['combined_features'] = df_filtered['listed_in'] + ' ' + df_filtered['description']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_filtered['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    recommendation = get_recommendations(selected_title, cosine_sim, df_filtered)
    # use recommendation ...

    if not recommendation.empty:
        st.markdown(f"🎯 You selected: `{selected_title}`")
        st.markdown("### 🎥 Recommended Titles:")

        for _, row in recommendation.iterrows():
            with st.container():
                st.markdown(f"**📌 Title**: {row['title']}")
                st.markdown(f"**📂 Genre**: {row['listed_in']}")
                st.markdown(f"**📝 Description**: {row['description'][:300]}...")
                st.markdown("---")
    else:
        st.markdown(f"❌ No recommendations found for `{selected_title}`. Please try another title.")
