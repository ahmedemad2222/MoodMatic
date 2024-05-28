import re
from transformers import pipeline
import pandas as pd
# Load the CSV file
df = pd.read_csv('/path')

# Display the first few rows to check the data
print(df.head()) 

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.strip()  # Remove leading/trailing whitespace
    return text

# Apply preprocessing to the 'text' column
df['cleaned_text'] = df['Description'].apply(preprocess_text)



# Load a pre-trained model for emotion detection
emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')

# Function to get the top emotion from the model's prediction
def get_top_emotion(text):
    predictions = emotion_classifier(text)
    top_emotion = max(predictions, key=lambda x: x['score'])
    return top_emotion['label']

# Apply emotion detection to the 'cleaned_text' column
df['emotion'] = df['cleaned_text'].apply(get_top_emotion)

# Display the first few rows with detected emotions
print(df[['Description', 'emotion']].head())
