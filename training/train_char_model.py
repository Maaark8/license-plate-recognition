import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import string
import os
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Model Definition (should be identical to _build_character_model in ocr_model.py) ---
CHARACTERS = string.ascii_uppercase + string.digits
NUM_CLASSES = len(CHARACTERS)
IMG_WIDTH, IMG_HEIGHT = 32, 32

def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Data Generation ---
def generate_char_image(char, font_path, size=(IMG_WIDTH, IMG_HEIGHT), augment=False):
    """Generates an image of a single character."""
    try:
        font = ImageFont.truetype(font_path, size=int(size[1] * 0.8)) # Adjust font size
    except IOError:
        print(f"Could not load font: {font_path}. Using default system font.")
        font = ImageFont.load_default()

    # Create a slightly larger canvas to allow for small augmentations
    canvas_size = (int(size[0]*1.2), int(size[1]*1.2))
    image = Image.new('L', canvas_size, color='black') # Black background
    draw = ImageDraw.Draw(image)

    # Get text size
    try: # For Pillow >= 9.2.0
        text_bbox = draw.textbbox((0,0), char, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except AttributeError: # For older Pillow
         w, h = draw.textsize(char, font=font) # Deprecated
         text_width, text_height = w, h


    # Position text in the center
    x = (canvas_size[0] - text_width) // 2
    y = (canvas_size[1] - text_height) // 2
    
    draw.text((x, y), char, fill='white', font=font) # White character

    if augment:
        # Simple augmentations (can be expanded)
        image = image.rotate(np.random.uniform(-5, 5), expand=False, fillcolor='black')
        # Add slight noise
        np_image = np.array(image)
        noise = np.random.randint(0, 15, np_image.shape, dtype=np.uint8) # Low noise
        np_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(np_image)

    # Crop back to original size (or center crop)
    left = (canvas_size[0] - size[0]) // 2
    top = (canvas_size[1] - size[1]) // 2
    right = left + size[0]
    bottom = top + size[1]
    image = image.crop((left, top, right, bottom))
    
    # Convert PIL image to OpenCV format (numpy array) and normalize
    cv_image = np.array(image)
    # cv_image = cv2.adaptiveThreshold(cv_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return cv_image.astype('float32') / 255.0


def create_dataset_from_font(font_path, characters_to_generate, samples_per_char=100):
    """Creates a dataset of character images and labels."""
    images = []
    labels = []
    char_to_num = {char: idx for idx, char in enumerate(characters_to_generate)}

    print(f"Generating dataset for {len(characters_to_generate)} characters, {samples_per_char} samples each...")
    for char_idx, char_val in enumerate(characters_to_generate):
        if (char_idx + 1) % 10 == 0 : print(f"  Generated for {char_idx+1}/{len(characters_to_generate)} characters...")
        for _ in range(samples_per_char):
            img = generate_char_image(char_val, font_path, augment=True)
            images.append(img)
            labels.append(char_to_num[char_val])
    
    print("Dataset generation complete.")
    return np.array(images).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1), np.array(labels)

# --- Main Training Script ---
if __name__ == '__main__':
    # CONFIGURATION
    FONT_PATH = "font/alte-din-1451-mittelschrift.regular.ttf"  # IMPORTANT: Replace with the actual path to your license plate font file (e.g., .ttf, .otf)
                             # If you don't have a specific one, Arial is a common fallback.
                             # On Linux, common paths: /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
                             # On Windows: C:/Windows/Fonts/Arial.ttf
    
    OUTPUT_MODEL_WEIGHTS_PATH = "../models/char_recognizer.weights.h5" # Relative to this script's location
                                                                  # Make sure 'models' dir exists at parent level

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_MODEL_WEIGHTS_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")


    # Generate data
    # You might need to download a font like 'DejaVuSans.ttf' or use a system one.
    # For example, on Ubuntu: sudo apt-get install fonts-dejavu-core
    # Check if your desired font_path exists, otherwise use a default or raise error.
    if not os.path.exists(FONT_PATH):
        print(f"Font file not found at {FONT_PATH}. Please provide a valid font path.")
        print("You can try common system fonts like 'arial.ttf' (Windows) or '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf' (Linux).")
        exit()
        
    images, labels = create_dataset_from_font(FONT_PATH, CHARACTERS, samples_per_char=200) # More samples are better

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, Labels shape: {y_val.shape}")

    # Build model
    model = build_model()
    model.summary()

    # Train model
    epochs = 30 # Increase for better results, monitor val_loss
    batch_size = 64
    
    # Optional: Add callbacks for early stopping and saving the best model
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
        # tf.keras.callbacks.ModelCheckpoint(filepath=OUTPUT_MODEL_WEIGHTS_PATH, monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=1)
        # Using EarlyStopping's restore_best_weights is often simpler than ModelCheckpoint for just weights.
    ]

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks)

    # Save the trained model weights
    model.save_weights(OUTPUT_MODEL_WEIGHTS_PATH)
    print(f"Trained model weights saved to {OUTPUT_MODEL_WEIGHTS_PATH}")

    # Evaluate
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Plot training history (optional)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()