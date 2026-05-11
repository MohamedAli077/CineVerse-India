import streamlit as st
import pandas as pd
import pickle
import requests

from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
from streamlit_searchbox import st_searchbox


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="CineVerse India",
    page_icon="🎬",
    layout="wide"
)


# ─────────────────────────────────────────────
# CSS  (minimal — only what's needed)
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Hero header */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .hero-accent { color: #e94560; }

    .hero-sub {
        font-size: 0.95rem;
        color: #999;
        margin-bottom: 0;
        line-height: 1.6;
    }
    .hero-sub b { color: #ccc; }

    /* Similarity badge */
    .sim-badge {
        display: inline-block;
        background: #1a1a2e;
        color: #e94560;
        border: 1px solid #e94560;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 6px 0 4px 0;
    }

    /* Movie title */
    .movie-title {
        font-size: 0.92rem;
        font-weight: 700;
        margin: 8px 0 2px 0;
        line-height: 1.3;
    }

    /* Meta row */
    .movie-meta {
        font-size: 0.80rem;
        color: #888;
        margin: 1px 0;
    }

    /* Section header */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        margin: 28px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e94560;
    }

    /* Card divider */
    .card-divider {
        border: none;
        border-top: 1px solid #1e1e1e;
        margin: 14px 0 2px 0;
    }

    /* Expander label tweak */
    details summary {
        font-size: 0.78rem !important;
        color: #888 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero-title">
    🎬 <span class="hero-accent">CineVerse</span> India
</div>
<p class="hero-sub">
    Find Indian movies similar to your favorites using
    <b>semantic recommendations</b> powered by
    <b>Transformer Embeddings</b> and <b>Cosine Similarity</b>.
</p>
""", unsafe_allow_html=True)

st.divider()


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    movies = pd.read_csv("data/final_indian_movies_dataset.csv")
    return movies

movies = load_data()

movies['movie_title'] = (
    movies['title'].astype(str)
    + ' ('
    + movies['year'].astype(str)
    + ')'
)


# ─────────────────────────────────────────────
# LOAD EMBEDDINGS
# ─────────────────────────────────────────────

@st.cache_resource
def load_embeddings():
    return pickle.load(open("embeddings.pkl", "rb"))

embeddings = load_embeddings()


# ─────────────────────────────────────────────
# INDICES
# ─────────────────────────────────────────────

indices = pd.Series(
    movies.index,
    index=movies['movie_title']
).drop_duplicates()


# ─────────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────────

movie_titles       = movies['movie_title'].tolist()
movie_titles_lower = [t.lower() for t in movie_titles]


def search_movies(searchterm: str):
    if not searchterm or len(searchterm.strip()) < 1:
        return []

    term       = searchterm.strip()
    term_lower = term.lower()

    prefix_matches = [
        movie_titles[i]
        for i, t in enumerate(movie_titles_lower)
        if t.startswith(term_lower)
    ]

    fuzzy_results = process.extract(
        term,
        movie_titles,
        scorer=fuzz.token_set_ratio,
        limit=15,
    )

    prefix_set   = set(prefix_matches)
    fuzzy_matches = [
        m[0] for m in fuzzy_results
        if m[0] not in prefix_set and m[1] >= 40
    ]

    combined = prefix_matches[:5] + fuzzy_matches
    seen, ordered = set(), []
    for title in combined:
        if title not in seen:
            seen.add(title)
            ordered.append(title)
        if len(ordered) >= 8:
            break

    return ordered


# ─────────────────────────────────────────────
# RECOMMEND
# ─────────────────────────────────────────────

def recommend(movie_title: str, n: int = 10):
    if not movie_title or movie_title not in indices:
        return []

    idx              = indices[movie_title]
    movie_embedding  = embeddings[idx]
    similarity_scores = cosine_similarity([movie_embedding], embeddings)[0]

    similar_movies = sorted(
        enumerate(similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []
    for movie_index, score in similar_movies:
        if movie_index == idx:
            continue

        row = movies.iloc[movie_index]

        # Pull cast + director if columns exist
        cast     = str(row['cast'])     if 'cast'     in row and str(row['cast'])     != 'nan' else None
        director = str(row['director']) if 'director' in row and str(row['director']) != 'nan' else None

        recommendations.append({
            'movie_title':      row['movie_title'],
            'genres':           str(row['genres'])   if str(row['genres'])   != 'nan' else None,
            'overview':         str(row['overview'])  if str(row['overview']) != 'nan' else None,
            'poster_url':       str(row['poster_url']),
            'rating':           row['rating'],
            'cast':             cast,
            'director':         director,
            'similarity_score': round(score * 100, 1),
        })

        if len(recommendations) >= n:
            break

    return recommendations


# ─────────────────────────────────────────────
# POSTER
# ─────────────────────────────────────────────

FALLBACK_POSTER = "https://placehold.co/300x450/1a1a2e/888888?text=No+Poster"

@st.cache_data(show_spinner=False)
def is_url_valid(url: str) -> bool:
    if not url or url == "nan":
        return False
    try:
        resp = requests.head(url, timeout=3, allow_redirects=True)
        return resp.status_code == 200
    except Exception:
        return False


def render_poster(url: str, title: str):
    use_url = url if is_url_valid(url) else FALLBACK_POSTER
    try:
        st.image(use_url, use_container_width=True)
    except Exception:
        st.image(FALLBACK_POSTER, use_container_width=True)


# ─────────────────────────────────────────────
# CONTROLS
# ─────────────────────────────────────────────

col_search, col_count = st.columns([3, 1])

with col_search:
    selected_movie = st_searchbox(
        search_movies,
        label="Search Movie",
        placeholder="Type a movie name…  e.g. RRR, 3 Idiots, Tumbbad",
        key="movie_search",
    )

with col_count:
    rec_count = st.select_slider(
        "Recommendations",
        options=[5, 10, 15, 20],
        value=10,
    )

run = st.button("🎬  Find Similar Movies")


# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────

if run and selected_movie:

    recommendations = recommend(selected_movie, n=rec_count)

    if not recommendations:
        st.warning(f"No recommendations found for **{selected_movie}**. Try another title.")

    else:
        st.markdown(
            f'<div class="section-header">'
            f'Similar to &nbsp;<span style="color:#e94560">{selected_movie}</span>'
            f'&nbsp;— Top {len(recommendations)} picks'
            f'</div>',
            unsafe_allow_html=True
        )

        COLS = 4
        cols = st.columns(COLS, gap="large")

        for i, movie in enumerate(recommendations):
            with cols[i % COLS]:

                # ── Poster ──
                render_poster(movie['poster_url'], movie['movie_title'])

                # ── Title ──
                st.markdown(
                    f'<div class="movie-title">{movie["movie_title"]}</div>',
                    unsafe_allow_html=True
                )

                # ── Similarity badge ──
                st.markdown(
                    f'<span class="sim-badge">🎯 {movie["similarity_score"]}% match</span>',
                    unsafe_allow_html=True
                )

                # ── Genre + Rating (always visible) ──
                if movie['genres']:
                    st.markdown(
                        f'<div class="movie-meta">🎭 {movie["genres"]}</div>',
                        unsafe_allow_html=True
                    )

                if movie['rating']:
                    st.markdown(
                        f'<div class="movie-meta">⭐ {movie["rating"]} / 10</div>',
                        unsafe_allow_html=True
                    )

                # ── View Details expander ──
                with st.expander("View Details"):

                    if movie['director']:
                        st.markdown(f"**🎬 Director:** {movie['director']}")

                    if movie['cast']:
                        st.markdown(f"**🎭 Cast:** {movie['cast']}")

                    if movie['genres']:
                        st.markdown(f"**📂 Genres:** {movie['genres']}")

                    st.markdown(f"**⭐ Rating:** {movie['rating']} / 10")

                    st.markdown(
                        f"**🎯 Similarity Score:** {movie['similarity_score']}%"
                    )

                    if movie['overview']:
                        st.markdown("**📖 Overview:**")
                        st.write(movie['overview'])

                st.markdown('<hr class="card-divider">', unsafe_allow_html=True)

elif run and not selected_movie:
    st.info("Please select a movie from the search box first.")