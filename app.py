from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

app = Flask(__name__)

# Sample Data for Coping Strategies (User Ratings)
coping_strategies_data = {
    "Meditation": [4, 5, 3, 2, 4,5],
    "Exercise": [5, 4, 2, 1, 3,1],
    "Journaling": [3, 4, 5, 2, 4,4],
    "Breathing Exercises": [4, 3, 4, 2, 4,4],
    "Creative Expression": [3, 3, 3, 3, 5,3]
}

# Convert the coping strategies data into a DataFrame (Users as rows)
df = pd.DataFrame(coping_strategies_data)

# Calculate cosine similarity matrix between users (each user is represented by a row)
similarity_matrix = cosine_similarity(df)

# Emotion to User Mapping (Mapping Emotional State to User Index)
emotion_to_user_mapping = {
    'anxiety': 0,      # User 1 - High anxiety
    'depression': 1,   # User 2 - Moderate anxiety
    'stress': 2,       # User 3 - Low anxiety
}

# Function to recommend coping strategies based on Collaborative Filtering
# Function to recommend coping strategies based on Collaborative Filtering
def recommend(user_index, similarity_matrix, df):
    similar_users = similarity_matrix[user_index]
    similar_users_idx = similar_users.argsort()[-3:][::-1]  # Top 3 most similar users
    recommended_strategies = df.iloc[similar_users_idx].mean(axis=0).round(1)  # Limit precision to 1 decimal place
    return recommended_strategies


# Home route to redirect to signup
@app.route('/')
def home():
    return redirect(url_for('signup'))

# Route for the signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Handle form submission (e.g., store user data)
        return redirect(url_for('about'))
    return render_template('signup.html')

# Route for the mental health support page (coping strategies info)
@app.route('/about')
def about():
    return render_template('about.html')

# Route for GAD-7 assessment page
@app.route('/gad7', methods=['GET', 'POST'])
def gad7():
    questions = [
        "1)Feeling nervous, anxious, or on edge",
        "2)Not being able to stop or control worrying",
        "3)Worrying too much about different things",
        "4)Trouble relaxing",
        "5)Being so restless that it's hard to sit still",
        "6)Becoming easily annoyed or irritable",
        "7)Feeling afraid, as if something awful might happen"
    ]
    if request.method == 'POST':
        scores = [int(request.form[f'q{i+1}']) for i in range(len(questions))]
        total_score = sum(scores)
        return redirect(url_for('result', score=total_score))
    
    return render_template('gad7.html', questions=questions)

# Route for results page after GAD-7 submission
@app.route('/result', methods=['GET', 'POST'])
def result():
    score = request.args.get('score', type=int)
    
    if score is not None:
        # Determine the emotional state based on GAD-7 score
        if score >= 15:
            recommendation = "Seek professional help."
            user_index = 0  # High anxiety
        elif score >= 10:
            recommendation = "Consider talking to a therapist."
            user_index = 1  # Moderate anxiety
        else:
            recommendation = "Minimal anxiety, keep practicing coping strategies."
            user_index = 2  # Low anxiety
        
        # Get AI-based recommended coping strategies
        recommended_strategies = recommend(user_index, similarity_matrix, df)
    else:
        recommendation = "No score provided."
        recommended_strategies = {}
    
    return render_template('result.html', score=score, recommendation=recommendation, recommended_strategies=recommended_strategies.to_dict())

# Route to handle AI recommendations through API (optional, if you want JSON)
@app.route('/recommend', methods=['POST'])
def recommend_coping_strategies():
    user_input = request.json.get('user_input')  # Emotion state from user (e.g., "anxiety")
    
    user_index = emotion_to_user_mapping.get(user_input.lower())
    if user_index is None:
        return jsonify({"error": "Invalid emotion state"}), 400

    recommended_strategies = recommend(user_index, similarity_matrix, df)
    return jsonify({"recommended_strategies": recommended_strategies.to_dict()})

if __name__ == '__main__':
    app.run(debug=True)
