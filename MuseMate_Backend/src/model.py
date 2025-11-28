# import os
# import pickle
# import warnings
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from PIL import Image, ImageFile
# import matplotlib.pyplot as plt

# # --- 1. CONFIGURATION ---
# # Allow PIL to load truncated images
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# # Silence warnings about potentially very large images
# warnings.simplefilter("ignore", Image.DecompressionBombWarning)
# # Disable TensorFlow oneDNN banner/noise
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# # --- Main Settings ---
# IMAGE_SIZE = (224, 224)
# BATCH_SIZE = 32
# EPOCHS = 30 # A model from scratch needs more epochs
# DATA_CSV = "data/CSV Files/master_artwork_info.csv"
# IMAGE_DIR = "data/images/train/train/"
# ARTIFACTS_DIR = "artifacts_scratch" # Directory to save plots, model, etc.
# ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
# ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "GIF", "BMP", "WEBP"}

# # --- 2. DATA LOADING AND CLEANING ---
# print("--- Starting Data Preparation ---")
# print("Loading CSV...")
# df = pd.read_csv(DATA_CSV)

# # Create full file paths
# df['filepath'] = df['filename'].apply(lambda x: os.path.join(IMAGE_DIR, x))

# # --- Robustly Filter Data ---
# # A) Remove rows pointing to missing files
# print(f"Initial row count: {len(df)}")
# df = df[df['filepath'].apply(os.path.exists)]
# print(f"Rows after removing missing file references: {len(df)}")

# # B) Remove rows with unreadable/corrupted images
# def is_image_loadable(path):
#     try:
#         # Fast extension gate first
#         _, ext = os.path.splitext(path)
#         if ext.lower() not in ALLOWED_EXTS:
#             return False
#         with Image.open(path) as img:
#             img.verify()  # Checks for file integrity
#             # Ensure format is one TF can decode
#             if getattr(img, 'format', None) not in ALLOWED_PIL_FORMATS:
#                 return False
#         return True
#     except Exception:
#         return False

# df = df[df['filepath'].apply(is_image_loadable)]
# print(f"Rows after removing corrupted images: {len(df)}")

# # C) Remove rare classes to ensure stratified split works and model can learn
# MIN_SAMPLES_PER_CLASS = 5
# df = df.dropna(subset=['title']) # Ensure title is not null
# title_counts = df['title'].value_counts()
# df = df[df['title'].map(title_counts) >= MIN_SAMPLES_PER_CLASS]
# print(f"Final rows after filtering rare titles: {len(df)}")

# # --- 3. LABEL ENCODING ---
# print("\n--- Encoding Labels ---")
# title_encoder = LabelEncoder()
# df['label'] = title_encoder.fit_transform(df['title'])
# num_classes = len(title_encoder.classes_)
# print(f"Number of unique classes to train: {num_classes}")

# # --- 4. DATA SPLITTING (Train/Validation only) ---
# print("\n--- Splitting Data ---")
# train_df, val_df = train_test_split(
#     df,
#     test_size=0.2, # 80% for training, 20% for validation
#     stratify=df['label'],
#     random_state=42
# )
# print(f"Training samples: {len(train_df)}")
# print(f"Validation samples: {len(val_df)}")

# # --- 5. DATA PIPELINE (tf.data) ---
# def process_path(file_path, label):
#     img = tf.io.read_file(file_path)
#     img = tf.image.decode_image(img, channels=3, expand_animations=False)
#     img = tf.image.resize(img, IMAGE_SIZE)
#     img.set_shape([*IMAGE_SIZE, 3])
#     img = img / 255.0
#     return img, label

# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = tf.data.Dataset.from_tensor_slices((train_df['filepath'].values, train_df['label'].values))
# train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# try:
#     train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
# except Exception:
#     train_ds = train_ds.ignore_errors()
# train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# val_ds = tf.data.Dataset.from_tensor_slices((val_df['filepath'].values, val_df['label'].values))
# val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# try:
#     val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
# except Exception:
#     val_ds = val_ds.ignore_errors()
# val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
# print("Data pipelines are ready.")

# # --- 6. BUILD MODEL FROM SCRATCH ---
# print("\n--- Building Model From Scratch ---")
# model = tf.keras.Sequential([
#     tf.keras.Input(shape=(*IMAGE_SIZE, 3)),
    
#     # Data Augmentation is crucial when training from scratch
#     tf.keras.layers.RandomFlip("horizontal"),
#     tf.keras.layers.RandomRotation(0.1),

#     # Convolutional Base
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.MaxPooling2D((2, 2)),

#     # Classifier Head
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(num_classes, activation='softmax')
# ])

# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )
# model.summary()

# # --- 7. TRAIN THE MODEL ---
# print("\n--- Starting Model Training ---")

# # Callbacks for smart training
# callbacks = [
#     # Stop training early if validation loss doesn't improve
#     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
#     # Save the best model automatically
#     tf.keras.callbacks.ModelCheckpoint(f"{ARTIFACTS_DIR}/scratch_model.keras", monitor='val_loss', save_best_only=True)
# ]

# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=EPOCHS,
#     callbacks=callbacks
# )

# # --- 8. SAVE ARTIFACTS AND PLOT RESULTS ---
# print("\n--- Training Complete ---")
# os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# # Save the label encoder
# with open(f"{ARTIFACTS_DIR}/title_encoder.pkl", "wb") as f:
#     pickle.dump(title_encoder, f)

# # Plot training history
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs_range = range(len(acc))

# plt.figure(figsize=(15, 6))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# plt.grid(True, alpha=0.3)

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.grid(True, alpha=0.3)

# plt.savefig(f'{ARTIFACTS_DIR}/training_history_scratch.png')
# print(f"Plots saved to {ARTIFACTS_DIR}/training_history_scratch.png")


# TRANSFER LEARNING 
import os
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image, ImageFile
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# --- Main Settings ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_CSV = "data/CSV Files/master_artwork_info.csv"
IMAGE_DIR = "data/images/train/train/"
ARTIFACTS_DIR = "artifacts_transfer" # New directory for this model
ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "GIF", "BMP", "WEBP"}

# --- Epochs for two-phase training ---
WARMUP_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
TOTAL_EPOCHS = WARMUP_EPOCHS + FINE_TUNE_EPOCHS

# --- 2. DATA LOADING AND CLEANING ---
print("--- Starting Data Preparation ---")
print("Loading CSV...")
df = pd.read_csv(DATA_CSV)
df['filepath'] = df['filename'].apply(lambda x: os.path.join(IMAGE_DIR, x))

print(f"Initial row count: {len(df)}")
df = df[df['filepath'].apply(os.path.exists)]
print(f"Rows after removing missing file references: {len(df)}")

def is_image_loadable(path):
    try:
        _, ext = os.path.splitext(path)
        if ext.lower() not in ALLOWED_EXTS: return False
        with Image.open(path) as img:
            img.verify()
            if getattr(img, 'format', None) not in ALLOWED_PIL_FORMATS: return False
        return True
    except Exception:
        return False

df = df[df['filepath'].apply(is_image_loadable)]
print(f"Rows after removing corrupted images: {len(df)}")

MIN_SAMPLES_PER_CLASS = 5
df = df.dropna(subset=['title'])
title_counts = df['title'].value_counts()
df = df[df['title'].map(title_counts) >= MIN_SAMPLES_PER_CLASS]
print(f"Final rows after filtering rare titles: {len(df)}")

# --- 3. LABEL ENCODING ---
print("\n--- Encoding Labels ---")
title_encoder = LabelEncoder()
df['label'] = title_encoder.fit_transform(df['title'])
num_classes = len(title_encoder.classes_)
print(f"Number of unique classes to train: {num_classes}")

# --- 4. DATA SPLITTING (Train/Validation only) ---
print("\n--- Splitting Data ---")
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# --- 5. DATA PIPELINE (tf.data) ---
def process_path(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img.set_shape([*IMAGE_SIZE, 3])
    # IMPORTANT: ResNet models expect inputs normalized from -1 to 1
    img = img / 127.5 - 1.0 
    return img, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = tf.data.Dataset.from_tensor_slices((train_df['filepath'].values, train_df['label'].values))
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
try:
    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
except Exception:
    train_ds = train_ds.ignore_errors()
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_df['filepath'].values, val_df['label'].values))
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
try:
    val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
except Exception:
    val_ds = val_ds.ignore_errors()
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
print("Data pipelines are ready.")

# --- 6. BUILD MODEL WITH TRANSFER LEARNING (ResNet50V2) ---
print("\n--- Building ResNet50V2 Transfer Learning Model ---")

# Define an explicit Input layer
inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")
x = data_augmentation(inputs)

# Load the ResNet50V2 base model
# We use input_shape because it's no longer the first layer
base_model = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_shape=(*IMAGE_SIZE, 3)
)

# Freeze the base model for the warmup phase
base_model.trainable = False

# Pass the augmented images through the base model
# training=False is important to keep BatchNormalization layers in inference mode
x = base_model(x, training=False) 

# Build the classifier head
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x) # Use a 50% dropout for this powerful model
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Create the final model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# --- 7. COMPILE AND TRAIN (PHASE 1: WARMUP) ---
print("\n--- Starting Model Warmup (Classifier Training) ---")

# Ensure artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Callbacks for smart training
checkpoint_path = f"{ARTIFACTS_DIR}/transfer_model.keras"
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
]

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history_warmup = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=WARMUP_EPOCHS,
    callbacks=callbacks
)

# --- 8. COMPILE AND TRAIN (PHASE 2: FINE-TUNING) ---
print("\n--- Starting Fine-Tuning ---")

# Unfreeze the base model
base_model.trainable = True

# Unfreeze the top 50 layers (ResNet has many layers)
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Re-compile the model with a very low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # 0.00001
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary() # Note the change in Trainable params

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=TOTAL_EPOCHS, # Train for the total number of epochs
    initial_epoch=history_warmup.epoch[-1] + 1, # Start from where we left off
    callbacks=callbacks # Reuse the same callbacks
)

# --- 9. SAVE ARTIFACTS AND PLOT RESULTS ---
print("\n--- Training Complete ---")

# Save the label encoder
with open(f"{ARTIFACTS_DIR}/title_encoder.pkl", "wb") as f:
    pickle.dump(title_encoder, f)
print("Label encoder saved.")

# Combine the history from both training phases
acc = history_warmup.history['accuracy'] + history_finetune.history['accuracy']
val_acc = history_warmup.history['val_accuracy'] + history_finetune.history['val_accuracy']
loss = history_warmup.history['loss'] + history_finetune.history['loss']
val_loss = history_warmup.history['val_loss'] + history_finetune.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.grid(True, alpha=0.3)

plt.savefig(f'{ARTIFACTS_DIR}/training_history_transfer.png')
print(f"Plots saved to {ARTIFACTS_DIR}/training_history_transfer.png")
print(f"Best model saved to {checkpoint_path}")