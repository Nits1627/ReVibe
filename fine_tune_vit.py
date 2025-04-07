#!/usr/bin/env python3
# fine_tune_vit.py

import tensorflow as tf
from transformers import TFViTModel, ViTImageProcessor
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
from PIL import Image
import random
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json

# Our interior design style classes
IDX_TO_CLASS = {
    0: "Transitional",
    1: "Modern",
    2: "Minimalist",
    3: "Industrial",
    4: "Coastal", 
    5: "Scandinavian",
    6: "Bohemian",
    7: "Mid-Century"
}

# Reverse mapping
CLASS_TO_IDX = {v.lower(): k for k, v in IDX_TO_CLASS.items()}

# Dataset path
DATASET_PATH = Path("sem6_atml_ds")

def create_model(model_name="google/vit-base-patch16-224", num_classes=8):
    """
    Create a custom model using the pre-trained ViT as a feature extractor.
    This avoids Lambda layers which can cause issues with deepcopy.
    """
    # Load pre-trained model and processor
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    # Create a simpler model - we'll extract features separately
    inputs = Input(shape=(768,), name="features_input")  # ViT outputs 768-dimensional vectors
    x = Dense(512, activation='relu')(inputs)
    x = Dropout(0.1)(x)
    outputs = Dense(num_classes, activation=None)(x)  # No activation for logits
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    return model, processor

def extract_features(images, model_name="google/vit-base-patch16-224"):
    """Extract features from images using the base ViT model."""
    # Load just the base ViT model
    base_model = TFViTModel.from_pretrained(model_name)
    
    # Process in batches to avoid memory issues
    features = []
    batch_size = 32
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        outputs = base_model(batch)[0][:, 0]  # Get the [CLS] token features
        features.append(outputs)
    
    return np.vstack(features)

def create_dataset(processor, batch_size=16, test_size=0.2, val_size=0.1):
    """
    Create dataset from our folder structure.
    Returns processed data for train, validation, and test sets.
    """
    # Lists to store processed data
    all_images = []
    all_labels = []
    
    # Process each style folder
    for style_folder in DATASET_PATH.iterdir():
        if not style_folder.is_dir():
            continue
            
        style_name = style_folder.name.lower()
        if style_name not in CLASS_TO_IDX:
            print(f"Warning: Folder {style_name} not in class mapping, skipping")
            continue
            
        label_idx = CLASS_TO_IDX[style_name]
        print(f"Processing {style_name} (class {label_idx})...")
        
        # Process each image in the folder
        image_files = list(style_folder.glob("*.jpg")) + list(style_folder.glob("*.jpeg")) + list(style_folder.glob("*.png"))
        for img_path in tqdm(image_files):
            try:
                # Open and preprocess the image
                image = Image.open(img_path).convert("RGB")
                
                # Process image with ViT processor
                inputs = processor(images=image, return_tensors="tf")
                pixel_values = inputs.pixel_values[0]  # Shape: [3, 224, 224]
                
                all_images.append(pixel_values)
                all_labels.append(label_idx)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays
    X = np.array(all_images)
    y = np.array(all_labels)
    
    # Split into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=val_size/(1-test_size),  # Adjust validation size
        stratify=y_train_val, 
        random_state=42
    )
    
    print(f"Train: {len(X_train)} images, Validation: {len(X_val)} images, Test: {len(X_test)} images")
    
    # Extract features using the ViT model
    print("Extracting features for training set...")
    X_train_features = extract_features(X_train)
    
    print("Extracting features for validation set...")
    X_val_features = extract_features(X_val)
    
    print("Extracting features for test set...")
    X_test_features = extract_features(X_test)
    
    return X_train_features, y_train, X_val_features, y_val, X_test_features, y_test

def fine_tune_model(model, X_train, y_train, X_val, y_val, epochs=10):
    """
    Fine-tune the classification model.
    """
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        batch_size=32,
        callbacks=[early_stopping]
    )
    
    # Save the final model weights (not whole model)
    os.makedirs("vit_finetuned_model", exist_ok=True)
    model.save_weights("vit_finetuned_model/model.weights.h5")
    
    return history, model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the fine-tuned model on the test set.
    """
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Get predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Class-wise accuracy
    for class_idx, class_name in IDX_TO_CLASS.items():
        class_mask = (y_test == class_idx)
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(y_pred[class_mask] == y_test[class_mask])
            print(f"  {class_name}: {class_accuracy:.4f} ({np.sum(class_mask)} samples)")
    
    return test_acc, y_pred

def save_model_for_inference(model, processor):
    """Save the model and metadata for inference."""
    # Create output directory
    os.makedirs('vit_finetuned_model', exist_ok=True)
    
    # Save the processor
    processor.save_pretrained('vit_finetuned_model')
    
    # Save the model weights in a more compatible format
    # Use the correct extension format required by Keras
    model.save_weights('vit_finetuned_model/model.weights.h5')
    
    # Save model architecture as JSON
    model_config = {
        "input_dim": 768,  # Feature dimension
        "hidden_dim": 512, # Hidden layer dimension
        "num_classes": 8,  # Number of output classes
    }
    
    with open('vit_finetuned_model/model_config.json', 'w') as f:
        json.dump(model_config, f)
    
    # Save the class mapping
    with open('vit_finetuned_model/class_mapping.txt', 'w') as f:
        for idx, class_name in IDX_TO_CLASS.items():
            f.write(f"{idx},{class_name}\n")
    
    print("Model saved for inference in 'vit_finetuned_model' directory")

if __name__ == "__main__":
    print("Starting ViT fine-tuning for interior design style classification")
    
    # Create the model and get the processor
    model, processor = create_model()
    print("Created classification model")
    
    # Process and prepare the dataset
    print("Processing dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = create_dataset(processor)
    
    # Fine-tune the model
    print("Starting fine-tuning...")
    history, fine_tuned_model = fine_tune_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate the model
    print("Evaluating the fine-tuned model...")
    evaluate_model(fine_tuned_model, X_test, y_test)
    
    # Save model for inference
    save_model_for_inference(fine_tuned_model, processor)
    
    print("Fine-tuning completed successfully!") 