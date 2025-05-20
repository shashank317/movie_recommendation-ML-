import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def prepare_data():
    df = pd.read_csv("data/netflix_titles.csv")
    df = df[['title', 'type', 'listed_in', 'description']].dropna()
    df['combined_features'] = df['listed_in'] + ' ' + df['description']
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return df, cosine_sim

def load_data(path='data/netflix_titles.csv'):
    df = pd.read_csv(path)
    df = df[['title', 'type', 'listed_in', 'description']].dropna()
    return df

def preprocess(df, content_type):
    df_filtered = df[df['type'] == content_type].reset_index(drop=True)
    df_filtered['combined_features'] = df_filtered['listed_in'] + ' ' + df_filtered['description']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_filtered['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return df_filtered, cosine_sim

def get_recommendations(title, cosine_sim, df_filtered):
    title_to_index = pd.Series(df_filtered.index, index=df_filtered['title'])
    if title not in title_to_index:
        return pd.DataFrame()  # or empty list
    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    indices = [i[0] for i in sim_scores]
    return df_filtered.iloc[indices][['title', 'type', 'listed_in', 'description']]

