# Hair Type and Scalp Analysis with U-Net for VSCode
# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Make sure TensorFlow is using GPU if available
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Define image dimensions
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

# Data paths
DATA_PATH = r"F:\KJSIM\prakruthi\Data\Images"
MASK_PATH = r"F:\KJSIM\prakruthi\Data\Masks"  # Create this directory if it doesn't exist
OUTPUT_DIR = r"F:\KJSIM\prakruthi\Output"      # Create this directory if it doesn't exist

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Auto-generate masks if they don't exist
def generate_masks_if_needed(data_path, mask_path):
    """
    Check if mask path exists and has masks corresponding to images
    If not, generate simple masks based on color thresholding
    """
    if not os.path.exists(mask_path):
        os.makedirs(mask_path, exist_ok=True)
        print(f"Created mask directory: {mask_path}")
    
    # Get list of images
    image_files = [f for f in os.listdir(data_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    existing_masks = [f for f in os.listdir(mask_path) if f.endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(mask_path) else []
    
    # Check if we need to generate masks
    if len(existing_masks) < len(image_files):
        print(f"Generating masks for {len(image_files) - len(existing_masks)} images...")
        
        for img_file in tqdm(image_files):
            # Check if mask already exists
            mask_file = f"mask_{os.path.splitext(img_file)[0]}.png"
            if mask_file not in existing_masks:
                # Load image
                img_path = os.path.join(data_path, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                
                # Convert to HSV for better color segmentation
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                # Define range for hair colors (adjust these values based on your images)
                # This is a simple starting point - may need tuning for your specific images
                lower_bound = np.array([0, 0, 0])
                upper_bound = np.array([180, 255, 90])  # Targets darker colors like hair
                
                # Create mask
                hair_mask = cv2.inRange(hsv, lower_bound, upper_bound)
                
                # Optional: Improve mask with morphological operations
                kernel = np.ones((5,5), np.uint8)
                hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
                hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
                
                # Save mask
                cv2.imwrite(os.path.join(mask_path, mask_file), hair_mask)
    else:
        print("All mask files exist - no need to generate new ones")

# Data loading and preprocessing functions
def load_data(data_path, mask_path, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    """
    Load images and masks, resize them to specified dimensions
    """
    images = []
    masks = []
    
    # Get all file names
    image_files = sorted([f for f in os.listdir(data_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Found {len(image_files)} images in {data_path}")
    
    for img_file in tqdm(image_files):
        # Determine mask filename
        mask_file = f"mask_{os.path.splitext(img_file)[0]}.png"
        mask_path_full = os.path.join(mask_path, mask_file)
        
        # Skip if mask doesn't exist
        if not os.path.exists(mask_path_full):
            print(f"Warning: No mask found for {img_file}, skipping...")
            continue
        
        # Read image and mask
        img = cv2.imread(os.path.join(data_path, img_file))
        if img is None:
            print(f"Warning: Could not read image {img_file}, skipping...")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_width, img_height))
        
        mask = cv2.imread(mask_path_full, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask {mask_file}, skipping...")
            continue
            
        mask = cv2.resize(mask, (img_width, img_height))
        
        # Normalize images to [0,1]
        img = img / 255.0
        mask = mask / 255.0
        
        images.append(img)
        masks.append(mask[..., np.newaxis])  # Add channel dimension
    
    return np.array(images), np.array(masks)

# U-Net Model Definition
def build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=1):
    """
    Build U-Net architecture
    """
    # Input
    inputs = Input(input_shape)
    
    # Encoder (Contracting Path)
    # Block 1
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Block 2
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Block 3
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Block 4
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bridge
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # Decoder (Expansive Path)
    # Block 6
    up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    # Block 7
    up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    # Block 8
    up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    # Block 9
    up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Output
    if num_classes == 1:  # Binary classification (e.g., hair vs non-hair)
        outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
    else:  # Multi-class segmentation (hair types, scalp conditions)
        outputs = Conv2D(num_classes, 1, activation='softmax')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Data augmentation function
def augment_data(imgs, masks, num_augmented=2):
    """
    Perform data augmentation on the fly
    """
    augmented_imgs = []
    augmented_masks = []
    
    for img, mask in zip(imgs, masks):
        # Add original image
        augmented_imgs.append(img)
        augmented_masks.append(mask)
        
        # Create augmented versions
        for i in range(num_augmented):
            # Random flip
            if np.random.random() > 0.5:
                img_aug = np.fliplr(img.copy())
                mask_aug = np.fliplr(mask.copy())
            else:
                img_aug = img.copy()
                mask_aug = mask.copy()
            
            # Random brightness adjustment
            brightness = np.random.uniform(0.8, 1.2)
            img_aug = np.clip(img_aug * brightness, 0, 1)
            
            # Random rotation (small angles to avoid cutting off too much)
            angle = np.random.uniform(-20, 20)
            M = cv2.getRotationMatrix2D((IMG_WIDTH/2, IMG_HEIGHT/2), angle, 1)
            
            img_aug = cv2.warpAffine(img_aug, M, (IMG_WIDTH, IMG_HEIGHT))
            mask_aug = cv2.warpAffine(mask_aug, M, (IMG_WIDTH, IMG_HEIGHT))
            
            # Make sure mask is binary
            mask_aug = (mask_aug > 0.5).astype(np.float32)
            
            # Add channel dimension back to mask if needed
            if len(mask_aug.shape) == 2:
                mask_aug = mask_aug[..., np.newaxis]
            
            augmented_imgs.append(img_aug)
            augmented_masks.append(mask_aug)
    
    return np.array(augmented_imgs), np.array(augmented_masks)

# Training function
def train_model(model, X_train, y_train, X_val, y_val, batch_size=8, epochs=50):
    """
    Train the U-Net model
    """
    # Callbacks
    model_path = os.path.join(OUTPUT_DIR, 'hair_scalp_unet_model.h5')
    early_stopping = EarlyStopping(patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, verbose=1)
    
    # Compile model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy' if model.output_shape[-1] == 1 else 'categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train model directly with arrays
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint]
    )
    
    return history, model_path

# Prediction and visualization
def predict_and_visualize(model, test_images, test_masks=None, num_samples=5, save_path=None):
    """
    Make predictions and visualize results
    """
    # Make predictions
    predictions = model.predict(test_images)
    
    # Plot results
    plt.figure(figsize=(15, 5*num_samples))
    
    for i in range(min(num_samples, len(test_images))):
        # Original image
        plt.subplot(num_samples, 3, i*3+1)
        plt.imshow(test_images[i])
        plt.title('Original Image')
        plt.axis('off')
        
        # Ground truth mask (if available)
        if test_masks is not None:
            plt.subplot(num_samples, 3, i*3+2)
            plt.imshow(test_masks[i].squeeze(), cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
        
        # Predicted mask
        plt.subplot(num_samples, 3, i*3+3)
        plt.imshow(predictions[i].squeeze(), cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
        
    plt.show()
    
    return predictions

# Hair type and scalp condition classification
def classify_hair_and_scalp(image, segmentation_mask):
    """
    Analyze the segmented hair and scalp to determine hair type and scalp conditions
    """
    # Extract hair region using the mask
    binary_mask = (segmentation_mask.squeeze() > 0.5).astype(np.uint8)
    hair_region = image * np.stack([binary_mask, binary_mask, binary_mask], axis=-1)
    
    # Calculate some real metrics from the image and mask
    # Hair density - percentage of the image covered by hair
    hair_density = np.mean(binary_mask)
    
    # Hair color analysis (simplified)
    hair_pixels = image[binary_mask > 0]
    if len(hair_pixels) > 0:
        mean_color = np.mean(hair_pixels, axis=0)
        brightness = np.mean(mean_color)
        # Contrast between channels can indicate hair color
        r_g_contrast = abs(mean_color[0] - mean_color[1])
        r_b_contrast = abs(mean_color[0] - mean_color[2])
        g_b_contrast = abs(mean_color[1] - mean_color[2])
    else:
        brightness = 0
        r_g_contrast = r_b_contrast = g_b_contrast = 0
    
    # Hair texture analysis - find hair "edges" to estimate curliness
    # More edges relative to hair area can indicate curlier hair
    if np.sum(binary_mask) > 0:
        edges = cv2.Canny(binary_mask, 50, 150)
        edge_density = np.sum(edges) / np.sum(binary_mask)
    else:
        edge_density = 0
    
    # Adjust these values to match reality better
    # Normalized edge density (higher = curlier)
    curliness = min(1.0, edge_density / 0.5)  
    
    # Thickness estimation (simplified)
    # We're using brightness as a proxy - darker hair often appears thicker
    # This is very simplified and would need refinement
    thickness = 1.0 - brightness if brightness > 0 else 0
    
    # Hair type analysis results
    hair_type_analysis = {
        "straightness": max(0, 1.0 - curliness),  # Inverse of curliness
        "curliness": curliness,
        "thickness": thickness,
        "density": hair_density
    }
    
    # Classify hair type based on analysis
    if hair_type_analysis["curliness"] < 0.2:
        hair_type = "Straight (Type 1)"
    elif hair_type_analysis["curliness"] < 0.5:
        hair_type = "Wavy (Type 2)"
    elif hair_type_analysis["curliness"] < 0.8:
        hair_type = "Curly (Type 3)"
    else:
        hair_type = "Coily (Type 4)"
    
    # Analyze scalp conditions (simplified)
    # In a real application, you would do more sophisticated analysis
    # Identify scalp areas (non-hair regions within a certain area)
    kernel = np.ones((20, 20), np.uint8)
    dilated_hair = cv2.dilate(binary_mask, kernel)
    potential_scalp = dilated_hair & ~binary_mask
    
    scalp_pixels = image[potential_scalp > 0]
    scalp_condition = {}
    
    if len(scalp_pixels) > 0:
        # Analyze scalp color
        mean_scalp_color = np.mean(scalp_pixels, axis=0)
        
        # Redness might indicate irritation
        redness = mean_scalp_color[0] - (mean_scalp_color[1] + mean_scalp_color[2])/2
        scalp_condition["irritation"] = max(0, min(1, redness * 5))
        
        # Brightness variance might indicate dryness/flaking
        brightness_var = np.var(np.mean(scalp_pixels, axis=1))
        scalp_condition["dryness"] = max(0, min(1, brightness_var * 10))
        
        # Yellowish tint might suggest oiliness
        yellowish = mean_scalp_color[1] - mean_scalp_color[2]
        scalp_condition["oiliness"] = max(0, min(1, yellowish * 5))
        
        # White specks might indicate dandruff
        high_brightness_pixels = np.mean(scalp_pixels > 0.8, axis=1)
        dandruff_indicator = np.mean(high_brightness_pixels)
        scalp_condition["dandruff"] = max(0, min(1, dandruff_indicator * 10))
    else:
        # Default values if no scalp is detected
        scalp_condition = {
            "dryness": 0,
            "oiliness": 0,
            "dandruff": 0,
            "irritation": 0
        }
    
    # Overall results
    results = {
        "Hair Type": hair_type,
        "Hair Properties": hair_type_analysis,
        "Scalp Conditions": scalp_condition
    }
    
    return results

# Real-world application functions
def load_and_predict_real_image(model, image_path, output_dir=None):
    """
    Load a real image, preprocess it, and make a prediction
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_normalized = img / 255.0  # Normalize
    
    # Make prediction
    mask_pred = model.predict(img_normalized[np.newaxis, ...])[0]
    
    # Analyze results
    analysis_results = classify_hair_and_scalp(img_normalized, mask_pred)
    
    # Visualize
    plt.figure(figsize=(12, 12))
    
    plt.subplot(221)
    plt.imshow(img_normalized)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(mask_pred.squeeze(), cmap='gray')
    plt.title('Hair Segmentation')
    plt.axis('off')
    
    # Overlay segmentation on original image
    plt.subplot(223)
    overlay = img_normalized.copy()
    hair_mask = mask_pred.squeeze() > 0.5
    overlay[hair_mask] = [0, 1, 0]  # Highlight hair in green
    plt.imshow(overlay)
    plt.title('Segmentation Overlay')
    plt.axis('off')
    
    # Show analysis results
    plt.subplot(224)
    plt.axis('off')
    plt.title('Analysis Results')
    
    # Create a text summary of the analysis
    result_text = f"Hair Type: {analysis_results['Hair Type']}\n\n"
    result_text += "Hair Properties:\n"
    for prop, value in analysis_results['Hair Properties'].items():
        result_text += f"  - {prop.capitalize()}: {value:.2f}\n"
    
    result_text += "\nScalp Conditions:\n"
    for cond, value in analysis_results['Scalp Conditions'].items():
        severity = "Low" if value < 0.3 else "Moderate" if value < 0.7 else "High"
        result_text += f"  - {cond.capitalize()}: {severity} ({value:.2f})\n"
    
    plt.text(0.1, 0.1, result_text, fontsize=9, verticalalignment='top')
    
    plt.tight_layout()
    
    # Save results if output_dir is provided
    if output_dir:
        img_name = os.path.basename(image_path)
        result_path = os.path.join(output_dir, f"analysis_{os.path.splitext(img_name)[0]}.png")
        plt.savefig(result_path)
        
        # Also save the mask
        mask_path = os.path.join(output_dir, f"mask_{os.path.splitext(img_name)[0]}.png")
        cv2.imwrite(mask_path, (mask_pred.squeeze() * 255).astype(np.uint8))
        
        print(f"Results saved to {result_path}")
    
    plt.show()
    
    return mask_pred, analysis_results

# Main execution function
def main():
    print("Starting hair and scalp analysis with U-Net...")
    
    # Check if we have masks; if not, generate them
    generate_masks_if_needed(DATA_PATH, MASK_PATH)
    
    # Load data
    print("Loading data from", DATA_PATH)
    images, masks = load_data(DATA_PATH, MASK_PATH)
    
    if len(images) == 0:
        print("No valid image/mask pairs found. Please check your paths and data.")
        return
    
    print(f"Loaded {len(images)} image-mask pairs.")
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
    
    print(f"Training data: {X_train.shape} images, {y_train.shape} masks")
    print(f"Validation data: {X_val.shape} images, {y_val.shape} masks")
    
    # Augment training data to improve generalization
    print("Augmenting training data...")
    X_train_aug, y_train_aug = augment_data(X_train, y_train, num_augmented=1)
    print(f"Augmented training data: {X_train_aug.shape} images, {y_train_aug.shape} masks")
    
    # Build U-Net model
    print("Building U-Net model...")
    model = build_unet()
    model.summary()
    
    # Train model
    print("Training model...")
    history, model_path = train_model(
        model, 
        X_train_aug, 
        y_train_aug, 
        X_val, 
        y_val,
        batch_size=4,  # Smaller batch size for memory constraints
        epochs=30
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Save training history plot
    history_plot_path = os.path.join(OUTPUT_DIR, 'training_history.png')
    plt.savefig(history_plot_path)
    print(f"Training history saved to {history_plot_path}")
    
    plt.show()
    
    # Make predictions on validation data
    print("Making predictions...")
    predict_and_visualize(
        model, 
        X_val[:5], 
        y_val[:5], 
        save_path=os.path.join(OUTPUT_DIR, 'validation_results.png')
    )
    
    # Analyze a sample image
    print("\nAnalyzing a sample image from validation set...")
    sample_image = X_val[0]
    sample_mask_pred = model.predict(sample_image[np.newaxis, ...])[0]
    
    analysis_results = classify_hair_and_scalp(sample_image, sample_mask_pred)
    
    print("\n=== Hair and Scalp Analysis Results ===")
    print(f"Hair Type: {analysis_results['Hair Type']}")
    print("\nHair Properties:")
    for prop, value in analysis_results['Hair Properties'].items():
        print(f"  - {prop.capitalize()}: {value:.2f}")
    
    print("\nScalp Conditions:")
    for cond, value in analysis_results['Scalp Conditions'].items():
        severity = "Low" if value < 0.3 else "Moderate" if value < 0.7 else "High"
        print(f"  - {cond.capitalize()}: {severity} ({value:.2f})")
    
    # Save the model and results
    print(f"\nModel saved to {model_path}")
    
    # Show code for using the model on new images
    code_example = f"""
# Example code to use the trained model on new images:
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('{model_path}')

# Function to analyze a new image
def analyze_hair_image(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img_normalized = img / 255.0
    
    # Make prediction
    mask_pred = model.predict(img_normalized[np.newaxis, ...])[0]
    
    # Visualize
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(img_normalized)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(mask_pred.squeeze(), cmap='gray')
    plt.title('Hair Segmentation')
    plt.axis('off')
    
    plt.subplot(133)
    overlay = img_normalized.copy()
    hair_mask = mask_pred.squeeze() > 0.5
    overlay[hair_mask] = [0, 1, 0]  # Highlight hair in green
    plt.imshow(overlay)
    plt.title('Segmentation Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return mask_pred

# Example usage
# analyze_hair_image('path/to/new/image.jpg')
"""
    
    # Save the example code to a file
    with open(os.path.join(OUTPUT_DIR, 'example_usage.py'), 'w') as f:
        f.write(code_example)
    
    print(f"Example usage code saved to {os.path.join(OUTPUT_DIR, 'example_usage.py')}")
    
    # Try the model on a few real images if they exist
    print("\nTrying the model on original images from the dataset...")
    image_files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    for i, img_file in enumerate(image_files[:3]):  # Test on first 3 images
        img_path = os.path.join(DATA_PATH, img_file)
        print(f"\nAnalyzing {img_file}...")
        try:
            mask_pred, analysis = load_and_predict_real_image(model, img_path, OUTPUT_DIR)
        except Exception as e:
            print(f"Error analyzing {img_file}: {e}")

if __name__ == "__main__":
    main()