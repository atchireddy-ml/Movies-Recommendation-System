import io
import re
import string
import textwrap
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).parent
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_MOVIE_URL = TMDB_BASE + "/movie/{movie_id}"
TMDB_SEARCH_URL = TMDB_BASE + "/search/movie"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "MovieRecommenderApp/1.0 (student project)"}

# >>> Paste your TMDB API key between the quotes below <<<
TMDB_API_KEY = "2ecff80a2c84cc0e89bc1b56cd3a0a8d"

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")


# ---------------------------------------------------------------- data loading
@st.cache_resource(show_spinner="Loading data...")
def load_data():
    """Load movie data and similarity matrix once, not on every interaction."""
    # reset_index: row POSITION must equal the row's position in the similarity matrix
    movies = pd.read_pickle(BASE_DIR / "movies.pkl").reset_index(drop=True)
    similarity = joblib.load(BASE_DIR / "similarity.joblib").astype(np.float32)
    if similarity.shape != (len(movies), len(movies)):
        raise ValueError(
            f"movies.pkl has {len(movies)} movies but similarity.joblib is "
            f"{similarity.shape[0]}x{similarity.shape[1]}. They must come from the same notebook run."
        )
    return movies, similarity


# ---------------------------------------------------------------- poster helpers
# Cached functions RAISE on failure, because Streamlit never caches an exception.
# That way a temporary failure (bad key, no internet) is retried next time instead of
# being remembered for 24 hours.
CACHE_TTL = 60 * 60 * 24
PLACEHOLDER_ID_START = 9_000_000   # ids at or above this were made up for Telugu rows without a TMDB id


@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def download_image(url: str) -> bytes:
    """Download an image on the server and return its bytes (raises if unusable).

    Doing this server-side means the poster still shows even if the user's browser
    or network blocks the image host. Bytes are verified to be a real image.
    """
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    Image.open(io.BytesIO(resp.content)).verify()   # raises if corrupt
    return resp.content


def make_placeholder(title: str) -> bytes:
    """Draw a poster-sized image with the movie title, so a poster is never empty."""
    width, height = 300, 450
    img = Image.new("RGB", (width, height), (28, 32, 48))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, width - 10, height - 10], outline=(200, 160, 60), width=3)
    try:
        font = ImageFont.load_default(size=26)
    except TypeError:  # very old Pillow
        font = ImageFont.load_default()
    wrapped = "\n".join(textwrap.wrap(string.capwords(title), width=14)[:6])
    draw.multiline_text(
        (width / 2, height / 2), wrapped, fill=(240, 240, 240),
        font=font, anchor="mm", align="center", spacing=8,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- metadata sources
@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def fetch_wikipedia_details(title: str, year: str = "") -> dict:
    """Fallback source: overview + poster URL from Wikipedia."""
    query = f"{title} {year} film".strip()
    resp = requests.get(
        WIKI_API_URL,
        params={
            "action": "query", "format": "json",
            "generator": "search", "gsrsearch": query, "gsrlimit": 1,
            "prop": "pageimages|extracts", "piprop": "thumbnail",
            "pithumbsize": 500, "exintro": 1, "explaintext": 1,
        },
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    if not pages:
        return {}
    page = next(iter(pages.values()))
    extract = (page.get("extract") or "").strip()
    if len(extract) > 700:
        extract = extract[:700].rsplit(" ", 1)[0] + "..."
    return {
        "overview": extract or None,
        "poster_url": (page.get("thumbnail") or {}).get("source"),
    }


@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def tmdb_movie(movie_id: int, api_key: str) -> dict:
    """Full TMDB record for one id (raises on any failure, so it is retried later)."""
    resp = requests.get(
        TMDB_MOVIE_URL.format(movie_id=movie_id),
        params={"api_key": api_key, "language": "en-US"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def find_tmdb_id(title: str, year: str, api_key: str):
    """Look a movie up by title (and year) when its stored id is not a real TMDB id.

    Returns the TMDB id, or None if there is no sensible match. Raises on network errors.
    """
    resp = requests.get(
        f"{TMDB_SEARCH_URL}",
        params={"api_key": api_key, "query": title, "include_adult": "false", "language": "en-US"},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])

    def year_gap(r):
        y = (r.get("release_date") or "")[:4]
        return abs(int(y) - int(year)) if (y.isdigit() and year.isdigit()) else 0

    # Telugu films first, then anything else; the year must agree to within 1
    for only_telugu in (True, False):
        pool = [r for r in results
                if (r.get("original_language") == "te" or not only_telugu) and year_gap(r) <= 1]
        if pool:
            return max(pool, key=lambda r: r.get("vote_count", 0))["id"]
    return None


@st.cache_data(show_spinner=False, ttl=600)
def check_api_key(api_key: str) -> bool:
    """True if TMDB accepts the key, False if it is rejected. Raises if TMDB is unreachable."""
    resp = requests.get(f"{TMDB_BASE}/configuration", params={"api_key": api_key}, timeout=10)
    if resp.status_code in (401, 403):
        return False
    resp.raise_for_status()
    return True


def safe(fn, *args):
    """Call a cached helper; return None instead of raising."""
    try:
        return fn(*args)
    except Exception:
        return None


def split_title_year(stored_title: str):
    """'avatar (2009)' / 'current (2009) [2]' -> ('avatar', '2009'). Undoes the notebook's suffixes."""
    t = re.sub(r"\s*\[\d+\]$", "", stored_title).strip()
    m = re.search(r"\s*\((\d{4})\)$", t)
    year = m.group(1) if m else ""
    return (t[: m.start()].strip() if m else t), year


def get_movie_details(movie_id: int, api_key: str, stored_title: str) -> dict:
    """Return title, year, overview and poster image bytes. Always includes a poster."""
    data = None
    if movie_id < PLACEHOLDER_ID_START:
        data = safe(tmdb_movie, movie_id, api_key)
    if not data:   # made-up id, or the id lookup failed: search TMDB by title instead
        name, year_hint = split_title_year(stored_title)
        found = safe(find_tmdb_id, name, year_hint, api_key)
        if found:
            data = safe(tmdb_movie, found, api_key)
    data = data or {}

    title = data.get("title") or string.capwords(split_title_year(stored_title)[0])
    year = (data.get("release_date") or "")[:4]
    overview = (data.get("overview") or "").strip() or None

    candidates = []
    if data.get("poster_path"):
        candidates.append(f"{POSTER_BASE_URL}{data['poster_path']}")

    if not overview or not candidates:
        wiki = safe(fetch_wikipedia_details, title, year) or {}
        overview = overview or wiki.get("overview")
        if wiki.get("poster_url"):
            candidates.append(wiki["poster_url"])

    if data.get("backdrop_path"):
        candidates.append(f"{POSTER_BASE_URL}{data['backdrop_path']}")

    poster = None
    for url in candidates:
        poster = safe(download_image, url)
        if poster:
            break
    if poster is None:
        poster = make_placeholder(title)   # last resort: never empty

    return {"title": title, "year": year, "overview": overview or "No overview available.", "poster": poster}


# ---------------------------------------------------------------- recommendation
def recommend(title: str, movies: pd.DataFrame, similarity: np.ndarray, n: int = 5):
    """Return the n most similar movies as a list of (row_index, score)."""
    # Row positions must stay aligned with the similarity matrix, so never drop rows.
    # If a title appears twice, the first match is used.
    idx = movies.index[movies["title"] == title][0]
    scores = similarity[idx]

    top = np.argpartition(scores, -(n + 1))[-(n + 1):]
    top = top[np.argsort(scores[top])[::-1]]
    top = [i for i in top if i != idx][:n]
    return [(int(i), float(scores[i])) for i in top]


def show_movie(row: pd.Series, score: float, api_key: str):
    """Render one recommendation: poster on the left, 'Title (Year)' + overview on the right."""
    details = get_movie_details(int(row["id"]), api_key, row["title"])
    year = f" ({details['year']})" if details["year"] else ""

    poster_col, text_col = st.columns([1, 2.5])
    with poster_col:
        st.image(details["poster"], width=200)
    with text_col:
        st.markdown(f"### {details['title']}{year}")
        st.write(details["overview"])
        st.caption(f"Match: {score:.0%}")
    st.divider()


# ---------------------------------------------------------------- UI
def main():
    st.title("🎬 Movie Recommendation System")

    # Tell the user straight away if posters cannot come from TMDB, instead of failing silently
    if TMDB_API_KEY == "PASTE_YOUR_TMDB_API_KEY_HERE":
        st.error("No TMDB API key set. Paste it into the TMDB_API_KEY line at the top of app.py.")
    else:
        try:
            if not check_api_key(TMDB_API_KEY):
                st.error("TMDB rejected your API key (check for typos or extra spaces in app.py).")
        except Exception:
            st.warning("Cannot reach TMDB right now (internet, firewall or ISP block). "
                       "Posters will fall back to Wikipedia or a drawn poster.")

    try:
        movies, similarity = load_data()
    except FileNotFoundError as e:
        st.error(f"Required data file not found: {e.filename}. Keep it in the same folder as app.py.")
        st.stop()
    except ValueError as e:
        st.error(f"Data files do not match. {e} Re-run the notebook's last cell to save both files together, "
                 "then copy BOTH into this folder (check OneDrive has finished syncing).")
        st.stop()

    titles = sorted(movies["title"].unique())
    selected = st.selectbox(
        "Choose a movie",
        titles,
        index=None,
        placeholder="Start typing a movie name...",
    )
    n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

    if st.button("Recommend", type="primary", disabled=selected is None):
        st.subheader("Recommended movies are:")
        with st.spinner("Fetching posters and details..."):
            for idx, score in recommend(selected, movies, similarity, n):
                show_movie(movies.iloc[idx], score, TMDB_API_KEY)


if __name__ == "__main__":
    main()
