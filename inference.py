import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import os
import argparse
import time

def load_model(model_path):
    """
    Loads the trained model from the specified path.
    
    Args:
        model_path: Path to the saved Keras model
        
    Returns:
        Loaded model
    """
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_new_data(file_path, segment_length=2.0, overlap=0.5):
    """
    Processes a new PhyPhox CSV file for inference.
    
    Args:
        file_path: Path to the CSV file
        segment_length: Length of each segment in seconds
        overlap: Overlap between adjacent segments (0-1)
        
    Returns:
        Array of processed segments ready for prediction
    """
    # CSV-Datei einlesen
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    # Spaltennamen überprüfen
    time_col = "Time (s)"
    feature_cols = [
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)"
    ]
    
    # Sicherstellen, dass die erforderlichen Spalten vorhanden sind
    for col in [time_col] + feature_cols:
        if col not in data.columns:
            print(f"Required column '{col}' not found in data: {file_path}")
            return None
    
    # Daten nach Zeit sortieren
    data = data.sort_values(by=time_col)
    
    # Abtastrate berechnen
    sampling_rate = 1 / np.median(np.diff(data[time_col]))
    print(f"Detected sampling rate: {sampling_rate:.2f} Hz")
    
    # Segmentierungsparameter
    stride = segment_length * (1 - overlap)
    total_time = data[time_col].max() - data[time_col].min()
    start_time = data[time_col].min()
    num_segments = max(1, int((total_time - segment_length) / stride) + 1)
    min_points = int(segment_length * sampling_rate * 0.8)
    
    print(f"Creating {num_segments} segments with min. {min_points} points per segment")
    
    segments = []
    segment_times = []  # Save start time of each segment for visualization
    
    for i in range(num_segments):
        segment_start = start_time + i * stride
        segment_end = segment_start + segment_length
        
        # Daten für dieses Segment extrahieren
        segment_data = data[(data[time_col] >= segment_start) & 
                           (data[time_col] < segment_end)]
        
        # Nur berücksichtigen, wenn genügend Datenpunkte vorhanden sind
        if len(segment_data) >= min_points:
            # Feature-Werte extrahieren
            segment_features = segment_data[feature_cols].values
            segments.append(segment_features)
            segment_times.append(segment_start)
    
    print(f"Extracted {len(segments)} valid segments from {file_path}")
    
    if not segments:
        print("No valid segments found. Check data and parameters.")
        return None, None
    
    # Standardize segment lengths
    lengths = [segment.shape[0] for segment in segments]
    target_length = int(np.median(lengths))
    print(f"Standardizing all segments to {target_length} time points")
    
    standardized_segments = []
    for segment in segments:
        if segment.shape[0] > target_length:
            standardized_segments.append(segment[:target_length])
        elif segment.shape[0] < target_length:
            padded = np.zeros((target_length, segment.shape[1]))
            padded[:segment.shape[0]] = segment
            standardized_segments.append(padded)
        else:
            standardized_segments.append(segment)
    
    X = np.array(standardized_segments)
    
    # Normalization
    original_shape = X.shape
    X_reshaped = X.reshape(-1, X.shape[-1])
    
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_reshaped)
    
    # Reshape back
    X = X_normalized.reshape(original_shape)
    
    print(f"Final data shape: {X.shape}")
    return X, np.array(segment_times)

def predict_activity(model, X):
    """
    Uses the model to predict activity classes for the provided segments.
    
    Args:
        model: Trained Keras model
        X: Preprocessed segment data
        
    Returns:
        Numpy array of class predictions and probabilities
    """
    # Get predictions
    pred_probas = model.predict(X)
    pred_classes = np.argmax(pred_probas, axis=1)
    
    return pred_classes, pred_probas

def visualize_predictions(file_path, segment_times, predictions, probabilities):
    """
    Visualizes the input data and corresponding predictions.
    
    Args:
        file_path: Path to the original CSV data
        segment_times: Start times of each segment
        predictions: Predicted classes for each segment
        probabilities: Prediction probabilities
    """
    # Load the original data for visualization
    data = pd.read_csv(file_path)
    time_col = "Time (s)"
    acc_cols = [
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)"
    ]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot the acceleration data
    for col in acc_cols:
        ax1.plot(data[time_col], data[col], label=col.split()[2])
    
    # Calculate the magnitude of acceleration for visualization
    mag = np.sqrt(data[acc_cols[0]]**2 + data[acc_cols[1]]**2 + data[acc_cols[2]]**2)
    ax1.plot(data[time_col], mag, 'k-', label='Magnitude', alpha=0.5)
    
    ax1.set_title('Acceleration Data and Predictions')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot the predictions
    class_names = ['Flat Ground', 'Stairs']
    colors = ['green', 'red']
    
    # Create a continuous prediction visualization
    min_time = data[time_col].min()
    max_time = data[time_col].max()
    
    # Plot prediction probabilities
    for i, time in enumerate(segment_times):
        if i < len(segment_times) - 1:
            segment_duration = segment_times[i+1] - time
        else:
            segment_duration = 2.0  # Assuming the last segment is 2 seconds
            
        pred_class = predictions[i]
        prob = probabilities[i][pred_class]
        
        ax2.barh(y=0, width=segment_duration, left=time, height=0.8, 
                color=colors[pred_class], alpha=0.6 + 0.4 * prob)
        
        # Add text label in the middle of the segment bar
        ax2.text(time + segment_duration/2, 0, 
                f"{class_names[pred_class]} ({prob:.2f})", 
                ha='center', va='center')
    
    ax2.set_xlim(min_time, max_time)
    ax2.set_yticks([])
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Activity Classification')
    
    # Add a legend for the predictions
    legend_patches = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
    ax2.legend(legend_patches, class_names, loc='upper right')
    
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.splitext(os.path.basename(file_path))[0] + "_predictions.png"
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")
    
    plt.show()

def real_time_simulation(model, file_path, segment_length=2.0, window_step=0.5):
    """
    Simulates real-time processing of the data by processing small windows sequentially.
    
    Args:
        model: Trained Keras model
        file_path: Path to the CSV file
        segment_length: Length of each segment in seconds
        window_step: Time step for advancing the window in seconds
    """
    # Load the data
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Required columns
    time_col = "Time (s)"
    feature_cols = [
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)"
    ]
    
    # Check columns
    for col in [time_col] + feature_cols:
        if col not in data.columns:
            print(f"Required column '{col}' not found in data: {file_path}")
            return
    
    # Sort data by time
    data = data.sort_values(by=time_col)
    
    # Get sampling rate
    sampling_rate = 1 / np.median(np.diff(data[time_col]))
    print(f"Detected sampling rate: {sampling_rate:.2f} Hz")
    
    # Calculate points per segment
    points_per_segment = int(segment_length * sampling_rate)
    
    # Get model input shape
    input_shape = model.layers[0].input_shape[0]
    expected_time_points = input_shape[1]
    
    print(f"Model expects {expected_time_points} time points per segment")
    
    # Initialize visualization
    plt.figure(figsize=(12, 6))
    plt.ion()  # Interactive mode on
    
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    
    line_x, = ax1.plot([], [], 'r-', label='X')
    line_y, = ax1.plot([], [], 'g-', label='Y')
    line_z, = ax1.plot([], [], 'b-', label='Z')
    line_mag, = ax1.plot([], [], 'k-', label='Magnitude', alpha=0.5)
    
    ax1.set_title('Real-time Acceleration Data')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    bar_container = ax2.bar(['Flat Ground', 'Stairs'], [0, 0], color=['green', 'red'])
    ax2.set_ylim(0, 1)
    ax2.set_title('Activity Probability')
    ax2.set_ylabel('Probability')
    
    plt.tight_layout()
    
    # Create a scaler for normalization
    scaler = StandardScaler()
    
    # Simulate real-time processing
    total_time = data[time_col].max() - data[time_col].min()
    start_time = data[time_col].min()
    end_time = data[time_col].max()
    
    print(f"Simulating real-time processing from {start_time:.1f}s to {end_time:.1f}s")
    print("Press Ctrl+C to stop the simulation")
    
    current_time = start_time
    
    try:
        while current_time + segment_length <= end_time:
            # Get data for the current window
            window_end = current_time + segment_length
            window_data = data[(data[time_col] >= current_time) & 
                              (data[time_col] < window_end)]
            
            # Extract features
            if len(window_data) >= points_per_segment * 0.8:  # At least 80% of expected points
                # Extract feature values
                features = window_data[feature_cols].values
                
                # Standardize the length
                if features.shape[0] > expected_time_points:
                    features = features[:expected_time_points]
                elif features.shape[0] < expected_time_points:
                    padded = np.zeros((expected_time_points, features.shape[1]))
                    padded[:features.shape[0]] = features
                    features = padded
                
                # Normalize
                features_flat = features.reshape(-1, features.shape[-1])
                features_norm = scaler.fit_transform(features_flat)
                features = features_norm.reshape(1, features.shape[0], features.shape[1])
                
                # Make prediction
                prediction = model.predict(features, verbose=0)
                pred_class = np.argmax(prediction[0])
                
                # Update visualization
                window_times = window_data[time_col].values
                
                line_x.set_data(window_times, window_data[feature_cols[0]])
                line_y.set_data(window_times, window_data[feature_cols[1]])
                line_z.set_data(window_times, window_data[feature_cols[2]])
                
                # Calculate and plot magnitude
                mag = np.sqrt(window_data[feature_cols[0]]**2 + 
                             window_data[feature_cols[1]]**2 + 
                             window_data[feature_cols[2]]**2)
                line_mag.set_data(window_times, mag)
                
                ax1.relim()
                ax1.autoscale_view()
                
                # Update the prediction bars
                for i, bar in enumerate(bar_container):
                    bar.set_height(prediction[0][i])
                
                # Highlight the predicted class
                for i, bar in enumerate(bar_container):
                    bar.set_alpha(1.0 if i == pred_class else 0.5)
                
                plt.draw()
                plt.pause(0.1)  # Small pause to allow GUI to update
                
                # Display prediction
                class_names = ['Flat Ground', 'Stairs']
                print(f"Time window {current_time:.1f}s - {window_end:.1f}s: "
                      f"Predicted {class_names[pred_class]} "
                      f"(Probability: {prediction[0][pred_class]:.2f})")
            else:
                print(f"Skipping window {current_time:.1f}s - {window_end:.1f}s: Not enough data points")
            
            # Move to the next window
            current_time += window_step
            
            # Simulate real-time processing delay
            time.sleep(window_step / 2)  # Sleep for half the window step time
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    plt.ioff()  # Turn off interactive mode
    plt.show()

def batch_process_directory(model, directory, segment_length=2.0, overlap=0.5):
    """
    Processes all PhyPhox CSV files in a directory and generates predictions.
    
    Args:
        model: Trained Keras model
        directory: Path to the directory containing CSV files
        segment_length: Length of each segment in seconds
        overlap: Overlap between adjacent segments (0-1)
    """
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv') and "Linear" in f]
    
    if not csv_files:
        print(f"No suitable CSV files found in {directory}")
        return
    
    print(f"Found {len(csv_files)} CSV files for processing")
    
    # Process each file
    for file in csv_files:
        file_path = os.path.join(directory, file)
        print(f"\nProcessing {file}...")
        
        # Process the data
        X, segment_times = process_new_data(file_path, segment_length, overlap)
        
        if X is not None:
            # Make predictions
            pred_classes, pred_probas = predict_activity(model, X)
            
            # Summarize results
            class_names = ['Flat Ground', 'Stairs']
            class_counts = {class_names[i]: np.sum(pred_classes == i) for i in range(len(class_names))}
            
            print("Prediction summary:")
            for cls, count in class_counts.items():
                percentage = (count / len(pred_classes)) * 100
                print(f"  {cls}: {count} segments ({percentage:.1f}%)")
            
            # Determine overall classification
            majority_class = class_names[np.argmax([class_counts[name] for name in class_names])]
            print(f"Overall classification: {majority_class}")
            
            # Visualize the results
            visualize_predictions(file_path, segment_times, pred_classes, pred_probas)

def main():
    parser = argparse.ArgumentParser(description='PhyPhox Activity Classification Inference')
    parser.add_argument('--model', default='phyphox_activity_classifier.keras',
                        help='Path to the trained model file')
    parser.add_argument('--file', help='Path to a single PhyPhox CSV file for inference')
    parser.add_argument('--dir', help='Path to a directory containing PhyPhox CSV files')
    parser.add_argument('--segment_length', type=float, default=2.0,
                        help='Length of each segment in seconds')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap between adjacent segments (0-1)')
    parser.add_argument('--realtime', action='store_true',
                        help='Simulate real-time processing')
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model)
    
    if model is None:
        print("Error: Could not load model")
        return
    
    # Process based on the provided arguments
    if args.file:
        if args.realtime:
            print(f"Starting real-time simulation for {args.file}")
            real_time_simulation(model, args.file, args.segment_length)
        else:
            print(f"Processing file: {args.file}")
            X, segment_times = process_new_data(args.file, args.segment_length, args.overlap)
            
            if X is not None:
                pred_classes, pred_probas = predict_activity(model, X)
                visualize_predictions(args.file, segment_times, pred_classes, pred_probas)
    elif args.dir:
        print(f"Processing all files in directory: {args.dir}")
        batch_process_directory(model, args.dir, args.segment_length, args.overlap)
    else:
        print("Error: Please provide either --file or --dir argument")
        print("Example usage:")
        print("  python inference.py --file LinearAcceleration_test.csv")
        print("  python inference.py --dir ./test_data/")
        print("  python inference.py --file LinearAcceleration_test.csv --realtime")

if __name__ == "__main__":
    main()
