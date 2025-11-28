import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
import random

# --- 1. SETUP ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
app = Flask(__name__, static_folder='../data/images') # <-- Point static folder to the images directory
CORS(app)

# --- 2. LOAD ALL ARTIFACTS ---
# (This section remains exactly the same)
print("Loading Keras model, label encoder, and master dataset...")
model = None
title_encoder = None
df_master = None

try:
    MODEL_PATH = 'artifacts_transfer/transfer_model.keras'
    model = tf.keras.models.load_model(MODEL_PATH)
    
    ENCODER_PATH = 'artifacts_transfer/title_encoder.pkl'
    with open(ENCODER_PATH, 'rb') as f:
        title_encoder = pickle.load(f)
        
    CSV_PATH = 'data/CSV Files/master_artwork_info.csv'
    df_master = pd.read_csv(CSV_PATH)
        
    print("All artifacts loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load artifacts. {e}")

# --- 3. DEFINE PREPROCESSING FUNCTION ---
# (This section remains exactly the same)
def preprocess_image(image_bytes):
    img = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    return img

# --- 4. CREATE API ENDPOINTS ---
@app.route('/')
def home():
    return "Flask AI Server for MuseMate is running!"

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model or not title_encoder or df_master is None:
        return jsonify({'error': 'Server artifacts not loaded.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if file:
        try:
            # --- MODEL PREDICTION ---
            image_bytes = file.read()
            processed_image = preprocess_image(image_bytes)
            predictions = model.predict(processed_image)
            
            predicted_id = np.argmax(predictions[0])
            predicted_title = title_encoder.inverse_transform([predicted_id])[0]
            confidence = float(np.max(predictions[0]))

            # --- DATA LOOKUP FOR THE MAIN ARTWORK ---
            artwork_info = df_master[df_master['title'] == predicted_title]
            
            if artwork_info.empty:
                return jsonify({'error': 'Predicted artwork not found in the database.'}), 404
            
            artwork_row = artwork_info.iloc[0]
            artist = artwork_row.get('artist', 'Unknown')
            style = artwork_row.get('style', 'Unknown')
            # ... (other fields)

            # --- HELPER FUNCTION TO CREATE FULL IMAGE URLS ---
            def create_image_url(filename):
                # Creates a URL like: http://localhost:5000/images/train/train/12345.jpg
                return url_for('static', filename=f'train/train/{filename}', _external=True)

            # --- 1. GET ARTIST'S TIMELINE (with full image_url) ---
            artist_timeline = []
            if artist != 'Unknown':
                artist_works_df = df_master[df_master['artist'] == artist].copy()
                artist_works_df['sortable_date'] = pd.to_numeric(artist_works_df['date'], errors='coerce')
                artist_works_df = artist_works_df.sort_values(by='sortable_date').dropna(subset=['sortable_date'])
                artist_works_df['image_url'] = artist_works_df['filename'].apply(create_image_url)
                artist_timeline = artist_works_df[['title', 'date', 'image_url']].to_dict('records')

            # --- 2. GET SIMILAR ARTWORKS (with full image_url) ---
            similar_artworks = []
            if style != 'Unknown':
                similar_df = df_master[(df_master['style'] == style) & (df_master['title'] != predicted_title)]
                num_samples = min(4, len(similar_df))
                if num_samples > 0:
                    sampled_artworks = similar_df.sample(n=num_samples)
                    sampled_artworks['image_url'] = sampled_artworks['filename'].apply(create_image_url)
                    similar_artworks = sampled_artworks[['title', 'artist', 'image_url']].to_dict('records')

            # --- 3. GET SIMILAR ARTISTS (with full image_url) ---
            similar_artists = []
            if style != 'Unknown':
                all_artists_in_style = df_master[df_master['style'] == style]['artist'].unique()
                other_artists_names = [a for a in all_artists_in_style if a != artist and isinstance(a, str) and len(a.split()) <= 4]
                num_samples = min(4, len(other_artists_names))
                if num_samples > 0:
                    selected_artists = random.sample(other_artists_names, num_samples)
                    for artist_name in selected_artists:
                        rep_artwork = df_master[df_master['artist'] == artist_name].sample(1).iloc[0]
                        similar_artists.append({
                            'artist_name': artist_name,
                            'image_url': create_image_url(rep_artwork['filename'])
                        })

            # --- RETURN FULL RESPONSE ---
            return jsonify({
                'predicted_title': predicted_title,
                'confidence': confidence,
                'artist': artist,
                'style': style,
                'image_url': create_image_url(artwork_row['filename']), # Add URL for the main image
                'artist_timeline': artist_timeline,
                'similar_artworks': similar_artworks,
                'similar_artists': similar_artists
                # ... (other fields)
            })
        except Exception as e:
            return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

# --- 5. RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)