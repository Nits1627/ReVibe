# classification.py

import tensorflow as tf
from transformers import TFViTForImageClassification, TFViTModel, ViTImageProcessor
from PIL import Image
import os
import numpy as np
from pathlib import Path
import json

def load_vit_model(model_name="google/vit-base-patch16-224", use_finetuned=True):
    """
    Load a pre-trained ViT model and its corresponding image processor from Hugging Face.
    If use_finetuned is True, load our fine-tuned model instead.
    """
    # Check if fine-tuned model exists
    finetuned_path = "vit_finetuned_model"
    
    if use_finetuned and os.path.exists(finetuned_path):
        print(f"Loading fine-tuned ViT model from {finetuned_path}")
        try:
            # Load the processor
            processor = ViTImageProcessor.from_pretrained(model_name)
            
            # Load model configuration
            model_config_path = os.path.join(finetuned_path, "model_config.json")
            if os.path.exists(model_config_path):
                with open(model_config_path, 'r') as f:
                    model_config = json.load(f)
                
                # Create feature extractor
                feature_extractor = TFViTModel.from_pretrained(model_name)
                
                # Create classifier model with proper input/output dimensions
                inputs = tf.keras.layers.Input(shape=(model_config["input_dim"],))
                x = tf.keras.layers.Dense(model_config["hidden_dim"], activation="relu")(inputs)
                x = tf.keras.layers.Dropout(0.1)(x)
                outputs = tf.keras.layers.Dense(model_config["num_classes"])(x)
                classifier = tf.keras.Model(inputs=inputs, outputs=outputs)
                
                # Load saved weights
                weights_path = os.path.join(finetuned_path, "model.weights.h5")
                if os.path.exists(weights_path):
                    try:
                        print(f"Loading model weights from {weights_path}")
                        classifier.load_weights(weights_path)
                        print("Loaded fine-tuned model weights")
                    except Exception as e:
                        print(f"Error loading model weights: {e}")
                        print("Creating model without pre-trained weights")
                
                # Create the combined model
                model = FeatureExtractorClassifier(feature_extractor, classifier)
                
                # Load class mapping
                if os.path.exists(os.path.join(finetuned_path, "class_mapping.txt")):
                    custom_idx_to_class = {}
                    with open(os.path.join(finetuned_path, "class_mapping.txt"), 'r') as f:
                        for line in f:
                            idx, class_name = line.strip().split(',')
                            custom_idx_to_class[int(idx)] = class_name
                    
                    # Update global mappings
                    global IDX_TO_CLASS, CLASS_TO_IDX
                    IDX_TO_CLASS = custom_idx_to_class
                    CLASS_TO_IDX = {v: k for k, v in IDX_TO_CLASS.items()}
                    print("Loaded custom class mapping")
                
                return model, processor
            else:
                # Fall back to pre-trained model
                print("Model configuration not found, loading pre-trained model instead")
                model = TFViTForImageClassification.from_pretrained(model_name)
                return model, processor
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            print("Falling back to pre-trained model")
            processor = ViTImageProcessor.from_pretrained(model_name)
            model = TFViTForImageClassification.from_pretrained(model_name)
            return model, processor
    else:
        print(f"Loading pre-trained ViT model from {model_name}")
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = TFViTForImageClassification.from_pretrained(model_name)
    return model, processor

# Class to combine feature extractor and classifier
class FeatureExtractorClassifier(tf.keras.Model):
    def __init__(self, feature_extractor, classifier):
        super(FeatureExtractorClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        
    def call(self, inputs, training=False):
        # Extract features from the input images
        features = self.feature_extractor(inputs)[0][:, 0]  # Get the [CLS] token
        # Apply classifier to the features
        return self.classifier(features, training=training)

# Our interior design style classes with proper mapping to the example filenames
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

# Reverse mapping for lookup
CLASS_TO_IDX = {v: k for k, v in IDX_TO_CLASS.items()}

# Map style names to example filenames in style_examples folder
STYLE_TO_FILENAME = {
    "Transitional": "trasitional_example.jpeg",  # Note the typo in the filename
    "Modern": "modern_example.jpeg",
    "Minimalist": "minimalist_example.jpeg",
    "Industrial": "industrialist_example.jpeg",
    "Coastal": "coastal_example.jpeg", 
    "Scandinavian": "scandanavian_example.jpeg",  # Note the typo in the filename
    "Bohemian": "bohemian_example.jpeg",
    "Mid-Century": "mid-century_example.jpeg"
}

def extract_features(image, model, processor):
    """Extract features from an image using the ViT model."""
    inputs = processor(image, return_tensors="tf")
    
    # Check if we have a custom combined model
    if isinstance(model, FeatureExtractorClassifier):
        # For FeatureExtractorClassifier, extract features from the feature extractor
        features = model.feature_extractor(inputs.pixel_values)[0][:, 0]
        return features.numpy()
    elif isinstance(model, tf.keras.Model) and not hasattr(model, 'vit'):
        # For standard Keras model
        features = model(inputs.pixel_values, training=False)
        return features.numpy()
    else:
        # For HuggingFace model
        features = model.vit(inputs.pixel_values).last_hidden_state[:, 0, :]
        return features.numpy()

def load_style_examples(model, processor):
    """Load and process style example images."""
    style_examples_dir = Path("style_examples")
    style_features = {}
    
    for style, filename in STYLE_TO_FILENAME.items():
        file_path = style_examples_dir / filename
        if file_path.exists():
            try:
                image = Image.open(file_path).convert('RGB')
                features = extract_features(image, model, processor)
                style_features[style] = features
            except Exception as e:
                print(f"Error processing {style} example: {e}")
    
    return style_features

# Global cache for style features to avoid recomputing
_STYLE_FEATURES_CACHE = None

def classify_style(image: Image.Image, model, processor):
    """
    Classify the uploaded image to determine its interior design style.
    
    If using a fine-tuned model, directly use the model's predictions.
    Otherwise, use the feature comparison approach with style examples.
    """
    global _STYLE_FEATURES_CACHE
    
    try:
        # Process the image
        inputs = processor(image, return_tensors="tf")
        
        # Check if we have a custom combined model or standard Keras model
        if isinstance(model, FeatureExtractorClassifier) or (isinstance(model, tf.keras.Model) and not hasattr(model, 'vit')):
            # Direct classification
            logits = model(inputs.pixel_values, training=False)
            predicted_class_idx = tf.argmax(logits, axis=1).numpy()[0]
            return IDX_TO_CLASS[predicted_class_idx]
            
        # Check if we're using a HuggingFace fine-tuned model
        elif hasattr(model, "config") and hasattr(model.config, "id2label") and model.config.id2label:
            # We have a fine-tuned classification model with proper labels
            # Use direct classification
            outputs = model(inputs.pixel_values)
            predicted_class_idx = tf.argmax(outputs.logits, axis=1).numpy()[0]
            
            # Map to style name using model's id2label mapping or our IDX_TO_CLASS
            if hasattr(model.config, "id2label") and predicted_class_idx in model.config.id2label:
                return model.config.id2label[predicted_class_idx]
            return IDX_TO_CLASS[predicted_class_idx]
        
        # If not a fine-tuned model, fall back to feature comparison
        # Initialize style features cache if not already done
        if _STYLE_FEATURES_CACHE is None:
            _STYLE_FEATURES_CACHE = load_style_examples(model, processor)
            
        # If we have no style examples, return a random style
        if not _STYLE_FEATURES_CACHE:
            random_idx = np.random.randint(0, len(IDX_TO_CLASS))
            return IDX_TO_CLASS[random_idx]
        
        # Extract features from the uploaded image
        image_features = extract_features(image, model, processor)
        
        # Compare with each style example
        similarities = {}
        for style, style_features in _STYLE_FEATURES_CACHE.items():
            # Calculate cosine similarity
            similarity = np.dot(image_features.flatten(), style_features.flatten()) / (
                np.linalg.norm(image_features) * np.linalg.norm(style_features))
            similarities[style] = similarity
        
        # Find the most similar style
        most_similar_style = max(similarities.items(), key=lambda x: x[1])[0]
        return most_similar_style
    
    except Exception as e:
        print(f"Error during classification: {e}")
        # Return a default style rather than "Unknown"
        return "Modern"

# Test the classification function if the script is run directly
if __name__ == "__main__":
    try:
        print("Loading ViT model...")
        model, processor = load_vit_model(use_finetuned=True)
        print("Model loaded successfully!")
        
        # Test with all style examples
        style_examples_dir = Path("style_examples")
        for style, filename in STYLE_TO_FILENAME.items():
            file_path = style_examples_dir / filename
            if file_path.exists():
                print(f"Testing with {style} example...")
                image = Image.open(file_path).convert('RGB')
                detected_style = classify_style(image, model, processor)
                print(f"  Detected style: {detected_style}")
        
        print("Available style classes:", IDX_TO_CLASS)
    except Exception as e:
        print(f"Error: {e}")