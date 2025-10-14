# Filename: app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Netflix Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Netflix Color Scheme & Custom CSS (Light Theme)
# -------------------------------
st.markdown("""
    <style>
    /* Netflix Light Theme */
    :root {
        --netflix-red: #E50914;
        --netflix-light-bg: #FFFFFF;
        --netflix-gray: #F3F3F3;
    }
    
    .stApp {
        background-color: #FFFFFF;
    }
    
    .main-header {
        background: linear-gradient(90deg, #E50914 0%, #B20710 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3em;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #F3F3F3 0%, #E8E8E8 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #E50914;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stSelectbox label, .stMultiSelect label {
        color: #E50914 !important;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    
    .sidebar-title {
        color: #E50914;
        font-size: 1.5em;
        font-weight: 700;
        margin-bottom: 20px;
    }
    
    h2, h3 {
        color: #000000 !important;
    }
    
    .recommendation-box {
        background: #F9F9F9;
        padding: 15px;
        border-radius: 8px;
        border-left: 3px solid #E50914;
        margin: 10px 0;
        color: #000000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #000000;
    }
    
    /* Text color */
    p, span, div {
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_netflix_titles.csv")
    df['description'] = df['description'].fillna('').astype(str)
    df['listed_in'] = df['listed_in'].fillna('').astype(str)
    df['title'] = df['title'].astype(str)
    df['country'] = df['country'].fillna('Unknown').astype(str)
    df['rating'] = df['rating'].fillna('Not Rated').astype(str)
    
    # Convert rating to star format
    rating_map = {
        'TV-Y': '⭐',
        'TV-Y7': '⭐⭐',
        'TV-G': '⭐⭐',
        'G': '⭐⭐',
        'TV-PG': '⭐⭐⭐',
        'PG': '⭐⭐⭐',
        'TV-14': '⭐⭐⭐⭐',
        'PG-13': '⭐⭐⭐⭐',
        'TV-MA': '⭐⭐⭐⭐⭐',
        'R': '⭐⭐⭐⭐⭐',
        'NC-17': '⭐⭐⭐⭐⭐',
        'NR': 'Not Rated',
        'UR': 'Not Rated',
        'Not Rated': 'Not Rated'
    }
    df['star_rating'] = df['rating'].map(rating_map).fillna('Not Rated')
    
    return df

df = load_data()

# TF-IDF for recommendations
@st.cache_resource
def build_recommender(df):
    df['text_features'] = df['description'] + " " + df['listed_in']
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['text_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_recommender(df)

# -------------------------------
# Recommendation Function
# -------------------------------
def get_recommendations(title, top_n=5):
    title = title.strip().lower()
    df['title_lower'] = df['title'].str.strip().str.lower()
    
    if title not in df['title_lower'].values:
        return None
    
    idx = df[df['title_lower'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    return df.iloc[top_indices][['title', 'type', 'star_rating', 'listed_in']]

# -------------------------------
# Header
# -------------------------------
st.markdown("""
    <div class="main-header">
        <h1>🎬 NETFLIX ANALYTICS DASHBOARD</h1>
    </div>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.markdown('<p class="sidebar-title">🎯 FILTERS</p>', unsafe_allow_html=True)

# Get unique genres
all_genres = []
for genres in df['listed_in'].str.split(','):
    all_genres.extend([g.strip() for g in genres])
unique_genres = sorted(list(set(all_genres)))

# Filters
selected_genres = st.sidebar.multiselect(
    "📂 Select Genre(s)",
    options=unique_genres,
    default=[]
)

selected_types = st.sidebar.multiselect(
    "🎭 Select Type",
    options=df['type'].unique(),
    default=[]
)

selected_ratings = st.sidebar.multiselect(
    "⭐ Select Rating",
    options=sorted(df['star_rating'].unique()),
    default=[]
)

# Apply filters
filtered_df = df.copy()

# Type filter
if selected_types:
    filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]

# Rating filter
if selected_ratings:
    filtered_df = filtered_df[filtered_df['star_rating'].isin(selected_ratings)]

# Genre filter
if selected_genres:
    filtered_df = filtered_df[filtered_df['listed_in'].apply(
        lambda x: any(genre in x for genre in selected_genres)
    )]

# -------------------------------
# Key Metrics
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 DASHBOARD STATS")
st.sidebar.metric("Total Titles", f"{filtered_df.shape[0]:,}")
st.sidebar.metric("Movies", f"{filtered_df[filtered_df['type']=='Movie'].shape[0]:,}")
st.sidebar.metric("TV Shows", f"{filtered_df[filtered_df['type']=='TV Show'].shape[0]:,}")

# -------------------------------
# Main Dashboard
# -------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("📺 Total Content", f"{filtered_df.shape[0]:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("🎬 Movies", f"{filtered_df[filtered_df['type']=='Movie'].shape[0]:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("📺 TV Shows", f"{filtered_df[filtered_df['type']=='TV Show'].shape[0]:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("🌍 Countries", f"{filtered_df['country'].nunique():,}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# Visualizations Row 1
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌍 Top 10 Countries Producing Netflix Content")
    top_countries = filtered_df['country'].value_counts().head(10)
    fig1 = px.bar(
        x=top_countries.values,
        y=top_countries.index,
        orientation='h',
        labels={'x': 'Number of Titles', 'y': 'Country'},
        color=top_countries.values,
        color_continuous_scale=['#E50914', '#B20710', '#8B0000']
    )
    fig1.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font_color='#000000',
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("🎭 Content Type Distribution")
    type_counts = filtered_df['type'].value_counts()
    fig2 = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        color_discrete_sequence=['#E50914', '#B20710'],
        hole=0.4
    )
    fig2.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font_color='#000000',
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Visualizations Row 2
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🍿 Top 10 Genres")
    filtered_df['genre_list'] = filtered_df['listed_in'].apply(lambda x: [i.strip() for i in x.split(',')])
    df_exploded = filtered_df.explode('genre_list')
    top_genres = df_exploded['genre_list'].value_counts().head(10)
    fig3 = px.bar(
        x=top_genres.values,
        y=top_genres.index,
        orientation='h',
        labels={'x': 'Number of Titles', 'y': 'Genre'},
        color=top_genres.values,
        color_continuous_scale=['#E50914', '#B20710', '#8B0000']
    )
    fig3.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font_color='#000000',
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("⭐ Rating Distribution")
    rating_counts = filtered_df['star_rating'].value_counts()
    fig4 = go.Figure(data=[go.Bar(
        x=rating_counts.index,
        y=rating_counts.values,
        marker_color='#E50914',
        text=rating_counts.values,
        textposition='auto',
    )])
    fig4.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font_color='#000000',
        xaxis_title='Rating',
        yaxis_title='Number of Titles',
        height=400
    )
    st.plotly_chart(fig4, use_container_width=True)

# -------------------------------
# Content Release Timeline
# -------------------------------
st.subheader("📅 Content Release Timeline")
if 'release_year' in filtered_df.columns:
    yearly_releases = filtered_df['release_year'].value_counts().sort_index()
    fig5 = px.area(
        x=yearly_releases.index,
        y=yearly_releases.values,
        labels={'x': 'Year', 'y': 'Number of Titles'},
        color_discrete_sequence=['#E50914']
    )
    fig5.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font_color='#000000',
        height=300
    )
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# -------------------------------
# Top Directors and Actors
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎬 Top 10 Directors")
    if 'director' in filtered_df.columns:
        # Split directors (handle multiple directors)
        directors_list = []
        for directors in filtered_df['director'].fillna('Unknown').astype(str):
            directors_list.extend([d.strip() for d in directors.split(',')])
        
        # Count and get top 10
        director_counts = pd.Series(directors_list).value_counts()
        director_counts = director_counts[director_counts.index != 'Unknown'].head(10)
        
        fig6 = px.bar(
            x=director_counts.values,
            y=director_counts.index,
            orientation='h',
            labels={'x': 'Number of Titles', 'y': 'Director'},
            color=director_counts.values,
            color_continuous_scale=['#E50914', '#B20710', '#8B0000']
        )
        fig6.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font_color='#000000',
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Director data not available in the dataset")

with col2:
    st.subheader("🌟 Top 10 Actors")
    if 'cast' in filtered_df.columns:
        # Split cast (handle multiple actors)
        actors_list = []
        for cast in filtered_df['cast'].fillna('Unknown').astype(str):
            actors_list.extend([c.strip() for c in cast.split(',')])
        
        # Count and get top 10
        actor_counts = pd.Series(actors_list).value_counts()
        actor_counts = actor_counts[actor_counts.index != 'Unknown'].head(10)
        
        fig7 = px.bar(
            x=actor_counts.values,
            y=actor_counts.index,
            orientation='h',
            labels={'x': 'Number of Titles', 'y': 'Actor'},
            color=actor_counts.values,
            color_continuous_scale=['#E50914', '#B20710', '#8B0000']
        )
        fig7.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font_color='#000000',
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.info("Cast data not available in the dataset")

st.markdown("---")

# -------------------------------
# Content Release Patterns
# -------------------------------
st.subheader("📆 When is Most Content Released?")

col1, col2 = st.columns(2)

with col1:
    if 'date_added' in filtered_df.columns:
        # Extract month from date_added
        filtered_df['date_added_clean'] = pd.to_datetime(filtered_df['date_added'], errors='coerce')
        filtered_df['month_added'] = filtered_df['date_added_clean'].dt.month_name()
        
        # Count by month
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        month_counts = filtered_df['month_added'].value_counts()
        month_counts = month_counts.reindex(month_order, fill_value=0)
        
        fig8 = px.bar(
            x=month_counts.index,
            y=month_counts.values,
            labels={'x': 'Month', 'y': 'Content Added'},
            color=month_counts.values,
            color_continuous_scale=['#E50914', '#B20710', '#8B0000'],
            title="Content Added by Month"
        )
        fig8.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font_color='#000000',
            showlegend=False,
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig8, use_container_width=True)
        
        # Find peak month
        peak_month = month_counts.idxmax()
        peak_count = month_counts.max()
        st.info(f"🔥 **Peak Month**: {peak_month} with {peak_count} titles added")
    else:
        st.warning("Date added information not available in the dataset")

with col2:
    if 'date_added' in filtered_df.columns and 'date_added_clean' in filtered_df.columns:
        # Extract day of week
        filtered_df['day_of_week'] = filtered_df['date_added_clean'].dt.day_name()
        
        # Count by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = filtered_df['day_of_week'].value_counts()
        day_counts = day_counts.reindex(day_order, fill_value=0)
        
        fig9 = go.Figure(data=[go.Bar(
            x=day_counts.index,
            y=day_counts.values,
            marker_color='#E50914',
            text=day_counts.values,
            textposition='auto',
        )])
        fig9.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font_color='#000000',
            xaxis_title='Day of Week',
            yaxis_title='Content Added',
            title="Content Added by Day of Week",
            height=400
        )
        st.plotly_chart(fig9, use_container_width=True)
        
        # Find peak day
        peak_day = day_counts.idxmax()
        peak_day_count = day_counts.max()
        st.info(f"📅 **Peak Day**: {peak_day} with {peak_day_count} titles added")
    else:
        st.warning("Date added information not available in the dataset")

st.markdown("---")

# -------------------------------
# Recommendation System
# -------------------------------
st.subheader("🔍 Find Similar Content")
st.write("Enter a movie or TV show title to get personalized recommendations")

col1, col2 = st.columns([4, 1])
with col1:
    title_input = st.text_input("Search for a title", placeholder="e.g., Stranger Things, The Crown, Breaking Bad", label_visibility="collapsed")
with col2:
    st.markdown("<div style='margin-top: 0px;'>", unsafe_allow_html=True)
    search_button = st.button("🔎 Search", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

if title_input and search_button:
    recs = get_recommendations(title_input)
    if recs is None:
        st.error("❌ Title not found in dataset. Please check the spelling and try again.")
    else:
        st.success(f"✅ Top 5 Recommendations for '{title_input}':")
        for idx, row in recs.iterrows():
            st.markdown(f"""
                <div class="recommendation-box">
                    <strong>🎬 {row['title']}</strong><br>
                    <em>Type:</em> {row['type']} | <em>Rating:</em> {row['star_rating']}<br>
                    <em>Genres:</em> {row['listed_in']}
                </div>
            """, unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>📊 Netflix Analytics Dashboard | Powered by Streamlit & Plotly</p>
    </div>
""", unsafe_allow_html=True)