# 🎬 Netflix Analytics Dashboard with Machine Learning

An advanced interactive **Streamlit Dashboard** that provides detailed insights into Netflix’s catalog of movies and TV shows, along with a **content recommendation system** and **machine learning models** to predict content ratings.  
This project combines **data visualization, recommendation systems, and ML classification** to offer a complete data analysis experience.

---

## 🚀 Key Features

### 📊 Dashboard Analytics
- Explore total movies and TV shows with interactive visuals  
- Visualize top countries producing Netflix content  
- Analyze genre and rating distributions  
- Study content release trends across years and months  
- Identify top directors and actors  

### 🔍 Recommendation System
- Search by **Title** to find similar shows or movies using **TF-IDF + Cosine Similarity**  
- Search by **Genre/Description** to find content matching specific moods or keywords  
- Display of detailed content information and interactive results  

### 🤖 Machine Learning Models
- Predict content **ratings** based on descriptions and genres  
- Models used:
  - 🌲 Random Forest Classifier  
  - 🎯 Support Vector Machine (SVM)  
  - 📊 Multinomial Naive Bayes  
- Performance comparison using **accuracy, confusion matrix, and classification metrics**

---

## 🧠 Tech Stack

- **Python 3.x**  
- **Streamlit** – Web app framework  
- **Pandas** – Data manipulation  
- **Plotly (Express & Graph Objects)** – Interactive visualizations  
- **Scikit-learn** – TF-IDF, Cosine Similarity, and ML models  
- **NumPy** – Numerical computations  

---

## 📂 Dataset

Dataset: **`cleaned_netflix_titles.csv`**

Contains:
- Title, Type (Movie/TV Show)  
- Director, Cast, Country  
- Release Year, Rating  
- Description and Genres  

The data is used for both **visual analytics** and **machine learning predictions**.

---

## 📸 Dashboard Sections

1. **Dashboard Page**
   <img width="1873" height="833" alt="image" src="https://github.com/user-attachments/assets/35beeb2b-5ea1-4dfe-9baf-4830ecd85382" />
 
2. **Recommendations Page** – Similar content search using NLP
   <img width="1866" height="869" alt="image" src="https://github.com/user-attachments/assets/d63df53e-f3d9-4e35-81ac-514a0a60db1a" />
   <img width="1870" height="853" alt="image" src="https://github.com/user-attachments/assets/dcd9d526-0341-49d5-8010-a7b455e43396" />

5. **ML Models Page** – Training results and accuracy comparison  
   <img width="1863" height="718" alt="image" src="https://github.com/user-attachments/assets/406adf22-c7ca-4c10-8b27-989556f90d4e" />
   <img width="1860" height="698" alt="image" src="https://github.com/user-attachments/assets/48ee75cb-5704-46a0-85bc-bcd41499898d" />
   <img width="1861" height="708" alt="image" src="https://github.com/user-attachments/assets/49ac079d-14a7-4dbc-bb3a-dc7d9d9282fa" />


---

## 📊 Insights

- TF-IDF vectorization converts movie descriptions and genres into numerical vectors.  
- Cosine similarity identifies closely related content titles.  
- Random Forest achieves the highest accuracy in rating prediction.  
- Visual dashboards enhance content analysis and discovery.  

---

## 🧾 License

This project is licensed under the **MIT License**.

---

## 👩‍💻 Author

**Pushti Shah**  
📧 pushtishah04@gmail.com  
🌐 [(https://github.com/PushtiShah04)]

> “Data + Visualization + ML = Smarter Insights on Entertainment.”
