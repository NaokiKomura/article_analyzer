from asari.api import Sonar
import pandas as pd
from tqdm import tqdm

# Load your CSV
df = pd.read_csv('data/classified_v4.csv')

# Filter data
filtered_df = df[df['分類ラベル'] == 1]

# Initialize Sonar
sonar = Sonar()

def analyze_sentiment(text):
    result = sonar.ping(text)
    return result['classes'][0]['confidence'] if result['top_class'] == 'positive' else result['classes'][1]['confidence']

# Apply sentiment analysis with tqdm for progress visualization
tqdm.pandas(desc="Analyzing Sentiments")
filtered_df['感情スコア'] = filtered_df['文'].progress_apply(analyze_sentiment)

# Normalize scores
min_score = filtered_df['感情スコア'].min()
max_score = filtered_df['感情スコア'].max()
filtered_df['正規化感情スコア'] = (filtered_df['感情スコア'] - min_score) / (max_score - min_score)

# Save to CSV
filtered_df.to_csv('data/classified_v5.csv', index=False)

