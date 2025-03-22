import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import gc
import tensorflow as tf

# Configure TensorFlow to use memory growth - this is critical
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

def read_phyphox_data(file_path, segment_length=5.0, overlap=0.5, class_label=None):
    """
    Process a single PhyPhox CSV file directly with memory-efficient operations
    """
    try:
        # Read only the columns we need
        time_col = "Time (s)"
        feature_cols = [
            "Linear Acceleration x (m/s^2)",
            "Linear Acceleration y (m/s^2)",
            "Linear Acceleration z (m/s^2)"
        ]
        data = pd.read_csv(file_path, usecols=[time_col] + feature_cols)
        
        # Sort by time
        data = data.sort_values(by=time_col)
        
        # Estimate sampling rate
        sampling_rate = 1 / np.median(np.diff(data[time_col]))
        
        # Calculate segment parameters
        stride = segment_length * (1 - overlap)
        total_time = data[time_col].max() - data[time_col].min()
        start_time = data[time_col].min()
        num_segments = max(1, int((total_time - segment_length) / stride) + 1)
        min_points = int(segment_length * sampling_rate * 0.8)
        
        segments = []
        labels = []
        
        # Extract segments
        for i in range(num_segments):
            segment_start = start_time + i * stride
            segment_end = segment_start + segment_length
            
            segment_data = data[(data[time_col] >= segment_start) & 
                               (data[time_col] < segment_end)]
            
            if len(segment_data) >= min_points:
                segment_features = segment_data[feature_cols].values
                segments.append(segment_features)
                labels.append(class_label)
        
        print(f"Extracted {len(segments)} valid segments from {file_path}")
        return segments, labels
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], []

def process_directory(directory, class_label, segment_length=1.0, overlap=0.5):
    """
    Process all CSV files in a directory with memory-efficient operations
    """
    all_segments = []
    all_labels = []
    
    for file_name in os.listdir(directory):
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory, file_name)
            segments, labels = read_phyphox_data(
                file_path, 
                segment_length=segment_length, 
                overlap=overlap, 
                class_label=class_label
            )
            all_segments.extend(segments)
            all_labels.extend(labels)
            
            # Explicitly release memory after processing each file
            gc.collect()
    
    if not all_segments:
        raise ValueError(f"No valid segments found in {directory}")
    
    return all_segments, all_labels

def standardize_segments(segments, labels):
    """
    Standardize segment lengths and convert to numpy arrays efficiently
    """
    # Find median length for standardization
    lengths = [segment.shape[0] for segment in segments]
    target_length = int(np.median(lengths))
    print(f"Standardizing all segments to {target_length} time points")
    
    # Pre-allocate the array for better memory efficiency
    num_segments = len(segments)
    num_features = segments[0].shape[1]
    X = np.zeros((num_segments, target_length, num_features))
    
    # Fill the pre-allocated array
    for i, segment in enumerate(segments):
        if segment.shape[0] > target_length:
            X[i, :, :] = segment[:target_length, :]
        else:
            X[i, :segment.shape[0], :] = segment
    
    y = np.array(labels)
    
    # Release memory
    del segments, labels
    gc.collect()
    
    return X, y

def normalize_features(X):
    """
    Normalize features with memory-efficient operations
    """
    # Store original shape
    original_shape = X.shape
    
    # Reshape for normalization
    X_reshaped = X.reshape(-1, X.shape[-1])
    
    # Normalize
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_reshaped)
    
    # Reshape back
    X = X_normalized.reshape(original_shape)
    
    # Release memory
    del X_reshaped, X_normalized
    gc.collect()
    
    return X

def create_model(input_shape, n_classes=2):
    """
    Create a smaller, memory-efficient model
    """
    # Reduced hyperparameters for memory efficiency
    head_size = 32  # Smaller attention heads
    num_heads = 2   # Fewer attention heads
    ff_dim = 32     # Smaller feed-forward network
    num_transformer_blocks = 1  # Fewer transformer blocks
    mlp_units = [32]  # Smaller MLP part
    dropout_rate = 0.5
    mlp_dropout = 0.5
    
    # Transformer encoder definition
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="sigmoid")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res
    
    # Model architecture
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(filters=16, kernel_size=3, padding="same", activation="relu")(inputs)
    
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout_rate)
    
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile with optimized settings
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["sparse_categorical_accuracy"]
    )
    
    return model

def train_and_evaluate(X, y, epochs=100, batch_size=16):
    """
    Train and evaluate the model with memory-efficient operations
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Memory cleanup
    gc.collect()
    
    # Create model
    model = create_model(input_shape=X_train.shape[1:])
    model.summary()
    
    # Callbacks with memory considerations
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_sparse_categorical_accuracy', 
            patience=15, 
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Save the model
    model.save("phyphox_activity_classifier.keras")
    
    return model, history

def main():
    # Directory paths
    class0_dir = "data/class0"  # Directory for class 0 (flat ground)
    class1_dir = "data/class1"  # Directory for class 1 (stairs)
    
    # Segmentation parameters
    segment_length = 20.0
    overlap = 0.7
    
    # Process data directories in memory-efficient way
    print("\n=== Processing Class 0 (flat ground) ===")
    segments_class0, labels_class0 = process_directory(
        class0_dir, 
        class_label=0, 
        segment_length=segment_length, 
        overlap=overlap
    )
    
    print("\n=== Processing Class 1 (stairs) ===")
    segments_class1, labels_class1 = process_directory(
        class1_dir, 
        class_label=1, 
        segment_length=segment_length, 
        overlap=overlap
    )
    
    # Combine segments and labels
    all_segments = segments_class0 + segments_class1
    all_labels = labels_class0 + labels_class1
    
    # Free memory
    del segments_class0, segments_class1, labels_class0, labels_class1
    gc.collect()
    
    # Standardize and normalize
    print("\n=== Standardizing segments ===")
    X, y = standardize_segments(all_segments, all_labels)
    
    print("\n=== Normalizing features ===")
    X = normalize_features(X)
    
    # Train model
    print("\n=== Starting model training ===")
    epochs = 200  # Reduced epochs with early stopping 
    batch_size = 16
    
    model, history = train_and_evaluate(X, y, epochs=epochs, batch_size=batch_size)
    print("Training complete. Model saved as 'phyphox_activity_classifier.keras'")
    
    return model, history

if __name__ == "__main__":
    main()
