# Filename: app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

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
    df['type'] = df['type'].fillna('').astype(str)
    
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

# -------------------------------
# TF-IDF for recommendations
# -------------------------------
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
        return None, None
    
    idx = df[df['title_lower'] == title].index[0]
    original_movie = df.iloc[idx]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    recommendations = df.iloc[top_indices]
    
    return original_movie, recommendations

# -------------------------------
# ML Model Training
# -------------------------------
@st.cache_resource
def train_ml_models(df):
    # Prepare data for classification (Predict Rating)
    rating_counts = df['rating'].value_counts()
    common_ratings = rating_counts[rating_counts >= 50].index
    df_ml = df[df['rating'].isin(common_ratings)].copy()
    
    # Create features
    df_ml['text_features'] = df_ml['description'] + " " + df_ml['listed_in']
    
    # TF-IDF Vectorization
    tfidf_ml = TfidfVectorizer(stop_words='english', max_features=1000)
    X = tfidf_ml.fit_transform(df_ml['text_features'])
    y = df_ml['rating']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_report = classification_report(y_test, rf_pred, output_dict=True, zero_division=0)
    rf_cm = confusion_matrix(y_test, rf_pred)
    
    # Train SVM
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    svm_report = classification_report(y_test, svm_pred, output_dict=True, zero_division=0)
    svm_cm = confusion_matrix(y_test, svm_pred)
    
    # Train Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    nb_report = classification_report(y_test, nb_pred, output_dict=True, zero_division=0)
    nb_cm = confusion_matrix(y_test, nb_pred)
    
    return {
        'models': {'rf': rf_model, 'svm': svm_model, 'nb': nb_model},
        'tfidf': tfidf_ml,
        'results': {
            'rf': {'accuracy': rf_accuracy, 'report': rf_report, 'cm': rf_cm, 'predictions': rf_pred},
            'svm': {'accuracy': svm_accuracy, 'report': svm_report, 'cm': svm_cm, 'predictions': svm_pred},
            'nb': {'accuracy': nb_accuracy, 'report': nb_report, 'cm': nb_cm, 'predictions': nb_pred}
        },
        'y_test': y_test,
        'classes': rf_model.classes_
    }

# Train models
with st.spinner('Training ML models...'):
    ml_results = train_ml_models(df)

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.markdown('<p class="sidebar-title">🎯 NAVIGATION</p>', unsafe_allow_html=True)
page = st.sidebar.radio("", ["📊 Dashboard", "🔍 Recommendations", "🤖 ML Models"], label_visibility="collapsed")

# -------------------------------
# DASHBOARD PAGE
# -------------------------------
if page == "📊 Dashboard":
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🎬 NETFLIX ANALYTICS DASHBOARD</h1>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Filters
    st.sidebar.markdown("---")
    st.sidebar.markdown('<p class="sidebar-title">🎯 FILTERS</p>', unsafe_allow_html=True)

    # Get unique genres
    all_genres = []
    for genres in df['listed_in'].str.split(','):
        all_genres.extend([g.strip() for g in genres])
    unique_genres = sorted(list(set(all_genres)))

    # Filters
    selected_genres = st.sidebar.multiselect("📂 Select Genre(s)", options=unique_genres, default=[])
    selected_types = st.sidebar.multiselect("🎭 Select Type", options=df['type'].unique(), default=[])
    selected_ratings = st.sidebar.multiselect("⭐ Select Rating", options=sorted(df['star_rating'].unique()), default=[])

    # Apply filters
    filtered_df = df.copy()

    if selected_types:
        filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]

    if selected_ratings:
        filtered_df = filtered_df[filtered_df['star_rating'].isin(selected_ratings)]

    if selected_genres:
        filtered_df = filtered_df[filtered_df['listed_in'].apply(
            lambda x: any(genre in x for genre in selected_genres)
        )]

    # Key Metrics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 DASHBOARD STATS")
    st.sidebar.metric("Total Titles", f"{filtered_df.shape[0]:,}")
    st.sidebar.metric("Movies", f"{filtered_df[filtered_df['type']=='Movie'].shape[0]:,}")
    st.sidebar.metric("TV Shows", f"{filtered_df[filtered_df['type']=='TV Show'].shape[0]:,}")

    # Main Dashboard Metrics
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

    # Visualizations Row 1
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

    # Visualizations Row 2
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
# RECOMMENDATIONS PAGE
# -------------------------------
elif page == "🔍 Recommendations":
    st.markdown("""
        <div class="main-header">
            <h1>🔍 CONTENT RECOMMENDATION SYSTEM</h1>
        </div>
    """, unsafe_allow_html=True)

    # Search Mode Selection
    search_mode = st.radio(
        "Choose Search Mode:",
        ["🎬 Search by Title", "🎭 Search by Genre/Description"],
        horizontal=True
    )

    st.markdown("---")

    # MODE 1: Search by Title (Original functionality)
    if search_mode == "🎬 Search by Title":
        st.write("### Find Similar Content Based on Title")
        #st.write("This system uses **TF-IDF** and **Cosine Similarity** to recommend content similar to your search.")

        col1, col2, col3 = st.columns([5, 2, 1])
        with col1:
            title_input = st.text_input("Search for a title", placeholder="e.g., Stranger Things, The Crown, Breaking Bad", label_visibility="collapsed")
        with col2:
            num_recs = st.slider("Recommendations", 3, 10, 5)
        with col3:
            st.markdown("<div style='margin-top: 0px;'>", unsafe_allow_html=True)
            search_button = st.button("🔎 Search", use_container_width=True, type="primary")
            st.markdown("</div>", unsafe_allow_html=True)

        if title_input and search_button:
            original_movie, recs = get_recommendations(title_input, top_n=num_recs)
            
            if original_movie is None:
                st.error("❌ Title not found in dataset. Please check the spelling and try again.")
            else:
                st.success(f"✅ Found: **{original_movie['title']}**")
                
                # Display original movie info
                st.markdown("### 📋 Original Content Details")
                st.markdown(f"""
                    <div class="recommendation-box">
                        <strong>🎬 {original_movie['title']}</strong><br>
                        <em>Type:</em> {original_movie['type']} | <em>Rating:</em> {original_movie['star_rating']}<br>
                        <em>Country:</em> {original_movie['country']}<br>
                        <em>Genres:</em> {original_movie['listed_in']}<br>
                        <em>Description:</em> {original_movie['description']}
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown(f"### 🎯 Top {num_recs} Similar Recommendations:")
                
                # Display recommendations
                for idx, row in recs.iterrows():
                    st.markdown(f"""
                        <div class="recommendation-box">
                            <strong>🎬 {row['title']}</strong><br>
                            <em>Type:</em> {row['type']} | <em>Rating:</em> {row['star_rating']}<br>
                            <em>Country:</em> {row['country']}<br>
                            <em>Genres:</em> {row['listed_in']}<br>
                            <em>Description:</em> {row['description'][:200]}{'...' if len(row['description']) > 200 else ''}
                        </div>
                    """, unsafe_allow_html=True)

    # MODE 2: Search by Genre/Description (NEW functionality)
    else:
        st.write("### Search Content by Genre, Mood, or Description")
        st.write("Enter keywords like **'comedy movie'**, **'horror'**, **'romantic drama'**, **'action thriller'**, etc.")

        col1, col2, col3 = st.columns([5, 2, 1])
        with col1:
            genre_input = st.text_input("Enter genre, mood, or description", 
                                       placeholder="e.g., comedy movie, horror, romantic drama, action thriller",
                                       label_visibility="collapsed")
        with col2:
            num_results = st.slider("Number of Results", 5, 20, 10)
        with col3:
            st.markdown("<div style='margin-top: 0px;'>", unsafe_allow_html=True)
            search_genre_button = st.button("🔎 Search", use_container_width=True, type="primary", key="genre_search")
            st.markdown("</div>", unsafe_allow_html=True)

        if genre_input and search_genre_button:
            # Create search query
            search_query = genre_input.lower().strip()
            
            # Search in genres and descriptions
            search_results = df[
                (df['listed_in'].str.lower().str.contains(search_query, na=False)) |
                (df['description'].str.lower().str.contains(search_query, na=False))
            ].head(num_results)
            
            if len(search_results) == 0:
                st.warning(f"⚠️ No results found for '{genre_input}'. Try different keywords like 'comedy', 'horror', 'action', etc.")
            else:
                st.success(f"✅ Found {len(search_results)} results for '{genre_input}':")
                
                # Display results
                for idx, row in search_results.iterrows():
                    st.markdown(f"""
                        <div class="recommendation-box">
                            <strong>🎬 {row['title']}</strong><br>
                            <em>Type:</em> {row['type']} | <em>Rating:</em> {row['star_rating']}<br>
                            <em>Country:</em> {row['country']}<br>
                            <em>Genres:</em> {row['listed_in']}<br>
                            <em>Description:</em> {row['description'][:250]}{'...' if len(row['description']) > 250 else ''}
                        </div>
                    """, unsafe_allow_html=True)
                
                # Show statistics
                st.markdown("---")
                st.markdown("### 📊 Search Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Results", len(search_results))
                with col2:
                    movies_count = len(search_results[search_results['type'] == 'Movie'])
                    st.metric("Movies", movies_count)
                with col3:
                    tv_count = len(search_results[search_results['type'] == 'TV Show'])
                    st.metric("TV Shows", tv_count)

# -------------------------------
# ML MODELS PAGE
# -------------------------------
elif page == "🤖 ML Models":
    st.markdown("""
        <div class="main-header">
            <h1>🤖 MACHINE LEARNING MODELS</h1>
        </div>
    """, unsafe_allow_html=True)

    st.write("### Rating Prediction Using Machine Learning")
    st.write("We trained three ML models to predict content ratings based on descriptions and genres:")

    # Model comparison
    st.subheader("📈 Model Accuracy Comparison")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🌲 Random Forest", f"{ml_results['results']['rf']['accuracy']:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎯 SVM", f"{ml_results['results']['svm']['accuracy']:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Naive Bayes", f"{ml_results['results']['nb']['accuracy']:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Accuracy chart
    accuracy_data = pd.DataFrame({
        'Model': ['Random Forest', 'SVM', 'Naive Bayes'],
        'Accuracy': [
            ml_results['results']['rf']['accuracy'],
            ml_results['results']['svm']['accuracy'],
            ml_results['results']['nb']['accuracy']
        ]
    })
    
    fig_acc = px.bar(accuracy_data, x='Model', y='Accuracy', 
                     title='Model Performance Comparison',
                     color='Accuracy',
                     color_continuous_scale=['#E50914', '#B20710', '#8B0000'])
    fig_acc.update_layout(
        yaxis_range=[0, 1],
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font_color='#000000',
        height=400
    )
    st.plotly_chart(fig_acc, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed model analysis
    st.subheader("🔬 Detailed Model Analysis")
    model_choice = st.selectbox("Select Model for Detailed Analysis", ["Random Forest", "SVM", "Naive Bayes"])
    
    model_map = {'Random Forest': 'rf', 'SVM': 'svm', 'Naive Bayes': 'nb'}
    selected_model = model_map[model_choice]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        st.write(f"#### Confusion Matrix - {model_choice}")
        cm = ml_results['results'][selected_model]['cm']
        classes = ml_results['classes']
        
        fig_cm = px.imshow(cm, 
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=classes,
                           y=classes,
                           color_continuous_scale='Reds',
                           title=f'Confusion Matrix')
        fig_cm.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font_color='#000000',
            height=500
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        # Classification Report
        st.write(f"#### Classification Metrics - {model_choice}")
        report = ml_results['results'][selected_model]['report']
        
        # Convert report to dataframe
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df[report_df.index.isin(classes)]
        
        fig_report = go.Figure(data=[
            go.Bar(name='Precision', x=report_df.index, y=report_df['precision'], marker_color='#E50914'),
            go.Bar(name='Recall', x=report_df.index, y=report_df['recall'], marker_color='#B20710'),
            go.Bar(name='F1-Score', x=report_df.index, y=report_df['f1-score'], marker_color='#8B0000')
        ])
        fig_report.update_layout(
            barmode='group',
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font_color='#000000',
            title='Metrics by Rating',
            height=500,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_report, use_container_width=True)
    
    # Model insights
    st.markdown("---")
    st.subheader("💡 Model Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="recommendation-box">
            <strong>🌲 Random Forest Classifier</strong><br><br>
            • Ensemble learning method<br>
            • Uses 100 decision trees<br>
            • Reduces overfitting<br>
            • Best for accuracy<br>
            • Handles high-dimensional data
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="recommendation-box">
            <strong>🎯 Support Vector Machine</strong><br><br>
            • Finds optimal hyperplane<br>
            • Linear kernel used<br>
            • Effective in high dimensions<br>
            • Good for text classification<br>
            • Clear margin separation
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="recommendation-box">
            <strong>📊 Naive Bayes Classifier</strong><br><br>
            • Probabilistic classifier<br>
            • Based on Bayes' theorem<br>
            • Assumes independence<br>
            • Fast and efficient<br>
            • Works well with TF-IDF
        </div>
        """, unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About")
st.sidebar.info("""
**Features:**
- 📊 Interactive visualizations
- 🔍 Content recommendations
- 🤖 ML models: RF, SVM, NB
- 📈 Model performance metrics
""")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>📊 Netflix Analytics Dashboard with ML | Powered by Streamlit & Plotly</p>
    </div>
""", unsafe_allow_html=True)