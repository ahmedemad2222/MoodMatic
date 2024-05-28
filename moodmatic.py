import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
file_path = 'Merged_Movies_Emotions.csv'
data = pd.read_csv(file_path)

# Data Cleaning
data = data.dropna()

# Feature Engineering
# Combine relevant text fields into a single feature
data['text'] = data['Reviews'] + ' ' + data['Description']

# Initialize the TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=5000)

# Fit and transform the text data into TF-IDF features
X = tfidf.fit_transform(data['text'])

# Encode the emotions as numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Emotion'])

# Model Training
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'random_forest_model.joblib'
joblib.dump(model, model_filename)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

# Recommendation System
def recommend_movie(emotion_input, data, model, tfidf, label_encoder):
    # Transform the input emotion to its numerical label
    try:
        emotion_label = label_encoder.transform([emotion_input])[0]
    except ValueError:
        return f"Emotion '{emotion_input}' is not recognized by the label encoder."

    # Filter movies with the given emotion
    recommended_movies = data[data['Emotion'] == emotion_input]['Movie_Name'].unique()

    # Debug: Check the number of movies found for the given emotion
    print(f"Number of movies found for emotion '{emotion_input}': {len(recommended_movies)}")

    if len(recommended_movies) > 0:
        return recommended_movies.tolist()
    else:
        return "No movies found for the given emotion."

# Example usage
emotion_input = 'sadness'
recommendations = recommend_movie(emotion_input, data, model, tfidf, label_encoder)

print(f"Number of recommendations: {len(recommendations)}")
print("Sample recommendations:", recommendations[:len(recommendations)])
