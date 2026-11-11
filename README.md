# 🎬 Movie Recommendation System

A content-based Movie Recommendation System that recommends movies based on their similarity to a selected movie. 
It uses machine learning techniques to identify similar movies and enhances the recommendation experience with movie posters and overviews.

## 🚀 Features

- **Content-Based Recommendations:** Recommends movies based on their content similarity.
- **Vectorization:** Converts movie-related textual data into numerical representations.
- **Cosine Similarity:** Uses cosine similarity to calculate similarity between movies.
- **Movie Posters:** Fetches movie posters using the TMDB API.
- **Movie Overviews:** Retrieves movie descriptions from Wikipedia and TMDB.
- **Interactive Experience:** Displays recommended movies along with their posters and descriptions.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Similarity Measurement:** Cosine Similarity
- **Data Processing:** Pandas, NumPy
- **API Integration:** TMDB API
- **Movie Information:** Wikipedia
- **Model Storage:** Joblib

## ⚙️ How It Works

1. Movie information is processed and converted into numerical vectors using vectorization.
2. Cosine similarity is calculated between movies to measure their content similarity.
3. When a user selects a movie, the system identifies movies with the highest similarity scores.
4. Recommended movies are displayed with their posters and overviews retrieved through TMDB and Wikipedia.

## 📁 Project Structure

```text
Movie-Recommendation-System/
│
├── app.py
├── similarity.joblib
├── movies.pkl
├── .gitignore
├── .gitattributes
├── requirements.txt
└── README.md
```

## 🔑 API Configuration

This project uses the TMDB API to retrieve movie posters and information.

1. Obtain an API key from [The Movie Database (TMDB)](https://www.themoviedb.org/).
2. Create a `.env` file in the project directory.
3. Add your API key:

```env
API_KEY=your_tmdb_api_key
```

4. Load the environment variable in Python:

```python
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
```

## 📦 Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Movie-Recommendation-System
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create your `.env` file and configure your TMDB API key.

## ▶️ Run the Project

Run the application using the appropriate command for your project.

For example, if your application uses Streamlit:

```bash
streamlit run app.py
```

## 🎯 Project Objective

The objective of this project is to build a movie recommendation system that provides relevant movie suggestions using content-based filtering while improving the user experience through API-based movie posters and descriptive overviews.

## 👨‍💻 Author

**Pammi Atchi Reddy**

- GitHub: [@atchireddy-ml](https://github.com/atchireddy-ml)
- LinkedIn: [Atchi Reddy](https://www.linkedin.com/in/atchi-reddy-pammi-3a53b134a/)
