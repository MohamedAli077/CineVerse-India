# 🎬 CineVerse India

A semantic Indian movie recommendation system built using Transformer Embeddings and Cosine Similarity.

## 🚀 Features

- Semantic movie recommendations
- Transformer-based embeddings
- Cosine similarity scoring
- Dynamic fuzzy search
- Streamlit interactive UI
- TMDB poster integration
- Curated Indian movie dataset

## 🛠️ Tech Stack

- Python
- Pandas
- Sentence Transformers
- Scikit-learn
- Streamlit
- RapidFuzz
- TMDB API

## 📌 How It Works

Movies are converted into semantic embeddings using:
- `all-MiniLM-L6-v2`

Recommendations are generated using:
- cosine similarity between embeddings

The system focuses on:
- storytelling style
- emotions
- themes
- narrative similarity

instead of simple genre matching.

## 🎥 Demo Recommendation Examples

- Tumbbad → Kantara, Virupaksha
- 3 Idiots → Happy Days, Chhichhore
- Jersey → Dear Comrade, Maidaan
- Hi Nanna → Sita Ramam, Ala Modalaindi

## ⚠️ Current Limitation

The current dataset contains around 200 curated Indian movies.  
Top recommendations are highly meaningful, but broader recommendation ranges can still improve with larger datasets.

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py