# Social-media-platform-
Social media platform description 
from transformers import pipeline
import pandas as pd

# Initialize emotion analysis pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Sample social media comments
comments = [
    "I am so happy with the way things turned out!",
    "This is frustrating and disappointing.",
    "I'm just okay, nothing special about today.",
    "I'm terrified about tomorrow's presentation.",
    "You made my day with your message!",
    "I hate how this app keeps crashing!"
]

# Analyze each comment
results = []
for comment in comments:
    emotions = emotion_classifier(comment)[0]
    top_emotion = max(emotions, key=lambda x: x['score'])
    results.append({
        "Comment": comment,
        "Predicted Emotion": top_emotion['label'],
        "Confidence": round(top_emotion['score'], 2)
    })

# Convert to DataFrame
df = pd.DataFrame(results)
print(df)
