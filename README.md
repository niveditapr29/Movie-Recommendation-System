# Movie-Recommendation-System
Movie recommendation system using NLP, TF-IDF and Cosine similarity.
Aim: This project aims to build a movie recommendation system using machine learning (ML) algorithms, specifically focusing on natural language processing (NLP) techniques like TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity. The system suggests top movie recommendations based on a given movie's features, such as genres, keywords, popularity, tagline, and director. This system uses Flask as a web framework to create a user-friendly interface.
Steps involved:
1.	Data Collection and Pre-Processing
The data used for this project is sourced from a CSV file (movies.csv), which contains details of various movies. The primary columns relevant for our recommendations are:
•	genres: The genre of the movie (e.g., action, comedy, etc.).
•	keywords: A set of keywords related to the movie’s plot and theme.
•	popularity: A numerical rating or score based on the movie's popularity.
•	tagline: A tagline or description for the movie.
•	director: The director of the movie.
To ensure the model performs optimally, we replace missing values with empty strings.
2.	Combining Features
We combine the selected features into a single text string for each movie. This concatenation helps capture the combined essence of a movie's characteristics, ensuring the machine learning model can use it effectively for comparison.
3.	Feature Vectorization
Using the TfidfVectorizer from sklearn, we convert the text data into numerical form (feature vectors). TF-IDF is used to capture the importance of each word in the context of the document (in this case, a movie), considering how frequently a word appears and how rare it is across the entire corpus.
4.	Calculating Cosine Similarity
Once the features are vectorized, we calculate the similarity between movies using cosine similarity. Cosine similarity measures the angle between two vectors (in this case, movie feature vectors). The smaller the angle, the more similar the two vectors (movies) are to each other.
5.	Recommendation Generation
When the user provides the name of a movie, we first search for the closest match using difflib.get_close_matches(). Once we identify the closest match, we fetch its index and retrieve the similarity scores of all other movies compared to it. These scores are sorted in descending order to find the most similar movies.
6.	Flask Web Interface
The recommendation system is integrated into a Flask web application. The user submits a movie name through a form, and the system responds with the top 10 movie recommendations.
