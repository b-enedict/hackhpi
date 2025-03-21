import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib  # For loading the saved scaler

def load_model(model_path="phyphox_activity_classifier.keras"):
    """
    Loads the trained model and scaler
    """
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_segment(segment_data, target_length, scaler):
    """
    Preprocesses segments exactly like in training
    """
    # Pad/truncate to target length
    if segment_data.shape[0] > target_length:
        segment_data = segment_data[:target_length]
    elif segment_data.shape[0] < target_length:
        padded = np.zeros((target_length, segment_data.shape[1]))
        padded[:segment_data.shape[0]] = segment_data
        segment_data = padded
    
    # Normalize with training scaler (reshape for proper transformation)
    original_shape = segment_data.shape
    segment_flat = segment_data.reshape(-1, 3)
    segment_normalized = scaler.transform(segment_flat)
    segment_data = segment_normalized.reshape(original_shape)
    
    # Add batch dimension
    return np.expand_dims(segment_data, axis=0)

def read_phyphox_file_for_inference(file_path, segment_length=1.0, overlap=0.5):
    """
    Matches the training data segmentation exactly
    """
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return [], []

    # Validate columns
    time_col = "Time (s)"
    feature_cols = [
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)"
    ]
    
    for col in [time_col] + feature_cols:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")

    data = data.sort_values(by=time_col)
    
    # Calculate segmentation parameters
    sampling_rate = 1 / np.median(np.diff(data[time_col]))
    stride = segment_length * (1 - overlap)
    total_time = data[time_col].max() - data[time_col].min()
    start_time = data[time_col].min()
    
    num_segments = max(1, int((total_time - segment_length) / stride) + 1)
    min_points = int(segment_length * sampling_rate * 0.8)

    segments = []
    segment_times = []
    
    for i in range(num_segments):
        segment_start = start_time + i * stride
        segment_end = segment_start + segment_length
        
        segment_data = data[(data[time_col] >= segment_start) & 
                           (data[time_col] < segment_end)]
        
        if len(segment_data) >= min_points:
            segment_features = segment_data[feature_cols].values
            segments.append(segment_features)
            segment_times.append((segment_start, segment_end))
    
    print(f"Extracted {len(segments)} segments from {file_path}")
    return segments, segment_times

def run_inference(model, file_path, segment_length=1.0, overlap=0.5):
    # Load scaler with better error handling
    scaler = None
    try:
        scaler = joblib.load('activity_scaler.save')
    except Exception as e:
        print(f"\nERROR: Could not load scaler - {e}")
        print("This usually means you need to:")
        print("1. Run the training code first")
        print("2. Ensure 'activity_scaler.save' is in the current directory")
        print("3. Verify file permissions")
        return None

    # Get segments
    segments, segment_times = read_phyphox_file_for_inference(
        file_path, segment_length, overlap
    )
    
    if not segments:
        return None

    # Get input shape from model
    try:
        target_length = model.input_shape[1]
        print(f"Using model input length: {target_length}")
    except Exception as e:
        print(f"Error getting input shape: {e}")
        return None

    # Preprocess all segments
    processed_segments = [
        preprocess_segment(seg, target_length, scaler) for seg in segments
    ]

    # Make predictions
    predictions = []
    for i, segment in enumerate(processed_segments):
        try:
            pred = model.predict(segment, verbose=0)
            class_idx = np.argmax(pred[0])
            confidence = np.max(pred[0])
            predictions.append({
                'segment_start': segment_times[i][0],
                'segment_end': segment_times[i][1],
                'predicted_class': class_idx,
                'confidence': confidence,
                'class_name': 'Flat Ground' if class_idx == 0 else 'Stairs/Ramp'
            })
        except Exception as e:
            print(f"Error processing segment {i}: {e}")

    # Create results dataframe
    results = pd.DataFrame(predictions)

    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Class predictions plot
    plt.subplot(2, 1, 1)
    plt.bar(results['segment_start'], results['predicted_class'], 
            width=(segment_length * (1-overlap)), align='edge',
            color=['green' if c == 0 else 'red' for c in results['predicted_class']])
    plt.yticks([0, 1], ['Flat Ground', 'Stairs/Ramp'])
    plt.title('Activity Classification Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Activity')

    # Confidence plot
    plt.subplot(2, 1, 2)
    plt.bar(results['segment_start'], results['confidence'], 
            width=(segment_length * (1-overlap)), align='edge',
            color=['lightgreen' if c == 0 else 'lightcoral' for c in results['predicted_class']])
    plt.title('Classification Confidence')
    plt.xlabel('Time (s)')
    plt.ylabel('Confidence')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig('activity_classification_results.png')
    plt.show()

    # Print summary
    print("\nClassification Summary:")
    print(results.groupby('class_name').agg({
        'confidence': 'mean',
        'segment_start': 'count'
    }).rename(columns={'segment_start': 'count'}))

    return results

def main():
    # Load model
    model = load_model()
    if model is None:
        return

    # Target file to analyze
    file_path = "data/Experiment_walk_inside/Linear Acceleration.csv"

    # Run inference with same parameters as training
    results = run_inference(
        model=model,
        file_path=file_path,
        segment_length=1.0,  # Must match training
        overlap=0.5         # Must match training
    )

    if results is not None:
        results.to_csv('inference_results.csv', index=False)
        print("Results saved to inference_results.csv")

if __name__ == "__main__":
    main()
