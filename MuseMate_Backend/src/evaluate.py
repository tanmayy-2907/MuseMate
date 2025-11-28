import os
import pickle
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageFile

# --- 1. CONFIGURATION ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_CSV = "data/CSV Files/all_data_info.csv" 
TEST_IMAGE_DIR = "data/images/test/" # Corrected path
ARTIFACTS_DIR = "artifacts_transfer" 

# --- 2. LOAD ARTIFACTS ---
print("--- Loading Model and Encoder ---")
try:
    MODEL_PATH = os.path.join(ARTIFACTS_DIR, "transfer_model.keras")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "title_encoder.pkl")
    with open(ENCODER_PATH, 'rb') as f:
        title_encoder = pickle.load(f)
        
    print("Model and encoder loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load artifacts: {e}")
    exit()

# --- 3. DATA PREPARATION FOR TEST SET ---
print("\n--- Preparing Test Dataset ---")
df_all = pd.read_csv(DATA_CSV)
df_test = df_all[df_all['in_train'] == False].copy()
df_test['filepath'] = df_test['new_filename'].apply(lambda x: os.path.join(TEST_IMAGE_DIR, x))

df_test = df_test[df_test['filepath'].apply(os.path.exists)]

def is_image_loadable(path):
    try:
        with Image.open(path) as img: img.verify()
        return True
    except Exception: return False
df_test = df_test[df_test['filepath'].apply(is_image_loadable)]

known_classes = set(title_encoder.classes_)
df_test = df_test[df_test['title'].isin(known_classes)]
df_test['label'] = title_encoder.transform(df_test['title'])
print(f"Found {len(df_test)} valid test images for known classes.")

# --- 4. DATA PIPELINE ---
def process_path(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img.set_shape([*IMAGE_SIZE, 3])
    img = img / 127.5 - 1.0 
    return img, label

AUTOTUNE = tf.data.AUTOTUNE
test_ds = tf.data.Dataset.from_tensor_slices((df_test['filepath'].values, df_test['label'].values))
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
try:
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
except Exception:
    test_ds = test_ds.ignore_errors()
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
print("Test data pipeline is ready.")

# --- 5. EVALUATE THE MODEL ---
print("\n--- Starting Final Evaluation ---")
results = model.evaluate(test_ds)
test_loss = results[0]
test_accuracy = results[1]

print("\n" + "="*30)
print("     FINAL MODEL REPORT CARD")
print("="*30)
print(f"      Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")
print("="*30)

# --- 6. PLOT FINAL TEST ACCURACY ---
print("\n--- Generating Final Test Accuracy Plot ---")

# Ensure artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Prepare data for the bar chart
labels = ['Test Accuracy']
scores = [test_accuracy]
colors = ['#59a14f'] # Green for the final score

plt.figure(figsize=(6, 6))
bars = plt.bar(labels, scores, color=colors)
plt.ylabel('Accuracy')
plt.title('Final Model Test Accuracy')
plt.ylim(0, 1.0) # Set y-axis from 0 to 1

# Add text label on top of the bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval*100:.2f}%', ha='center', va='bottom')

plot_path = os.path.join(ARTIFACTS_DIR, 'test_accuracy_plot.png')
plt.savefig(plot_path)
print(f"Final test accuracy plot saved to {plot_path}")

