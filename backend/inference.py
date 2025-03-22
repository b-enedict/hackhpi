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
    Load a Keras model from the specified path.
    
    Parameters:
    model_path (str): Path to the model file
    
    Returns:
    model: Loaded Keras model or None if loading fails
    """
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_sensor_input(model, sensor_data, segment_length=20, overlap=0.0):
    """
    Process sensor input and make predictions using the loaded model.
    
    Parameters:
    model: Loaded Keras model
    sensor_data: Tuple of (x, y, z) acceleration data
    segment_length (int): Length of each segment for prediction
    overlap (float): Overlap between segments (0-1)
    
    Returns:
    int: Predicted class (0 for flat ground, 1 for stairs)
    """
    x_data, y_data, z_data = sensor_data
    
    # Ensure data is in numpy array format
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    z_data = np.array(z_data)
    
    # Check if we have enough data for a segment
    if len(x_data) < segment_length:
        print(f"Warning: Input data length ({len(x_data)}) is less than segment_length ({segment_length})")
        # Pad with zeros if needed
        padding_length = segment_length - len(x_data)
        x_data = np.pad(x_data, (0, padding_length), 'constant')
        y_data = np.pad(y_data, (0, padding_length), 'constant')
        z_data = np.pad(z_data, (0, padding_length), 'constant')
    
    # Combine the axes data
    features = np.column_stack((x_data, y_data, z_data))
    
    # Make sure the data matches the model's expected input shape
    input_shape = model.layers[0].input_shape[0]
    expected_time_points = input_shape[1] if len(input_shape) > 1 else segment_length
    
    if features.shape[0] > expected_time_points:
        features = features[:expected_time_points]
    elif features.shape[0] < expected_time_points:
        padded = np.zeros((expected_time_points, features.shape[1]))
        padded[:features.shape[0]] = features
        features = padded
    
    # Normalize the data
    scaler = StandardScaler()
    features_flat = features.reshape(-1, features.shape[-1])
    features_norm = scaler.fit_transform(features_flat)
    features = features_norm.reshape(1, features.shape[0], features.shape[1])
    
    # Make prediction
    prediction = model.predict(features, verbose=0)
    pred_class = np.argmax(prediction[0])
    
    return pred_class

def acceleration_data(x, y, z, time, model_path="best_model.keras"):
    """
    Process acceleration data and return activity labels with time segments.
    
    Parameters:
    x (list): X-axis acceleration data
    y (list): Y-axis acceleration data
    z (list): Z-axis acceleration data
    time (float): Start time of the data sequence
    model_path (str): Path to the model file
    
    Returns:
    tuple: (labels, starts, ends) lists containing activity labels and time segments
    """
    segment_length = 20
    
    # Load the model
    model = load_model(model_path)
    if model is None:
        return [], [], []
    
    labels = []
    starts = []
    ends = []
    
    # Convert lists to numpy arrays for easier processing
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    
    # Calculate number of complete segments
    num_segments = len(x) // segment_length
    
    start = time
    end = time
    
    for i in range(num_segments):
        start = end
        
        segment_start_idx = i * segment_length
        segment_end_idx = segment_start_idx + segment_length
        
        # Extract segment data
        segment_x = x[segment_start_idx:segment_end_idx]
        segment_y = y[segment_start_idx:segment_end_idx]
        segment_z = z[segment_start_idx:segment_end_idx]
        
        # Process this segment
        sensor_data = (segment_x, segment_y, segment_z)
        prediction = process_sensor_input(model, sensor_data, segment_length, 0.0)
        
        # Assuming some time interval for each data point (e.g., 0.01s per reading)
        # This could be adjusted based on actual sampling rate
        time_per_segment = 0.01 * segment_length
        end = start + time_per_segment
        
        # Assign label based on prediction
        if prediction == 1:
            labels.append("stair")
        elif prediction == 0:
            labels.append("no_stairs")
        
        starts.append(start)
        ends.append(end)
    
    return labels, starts, ends

def visualize_results(x, y, z, time, labels, starts, ends):
    """
    Visualize the acceleration data and activity classifications.
    
    Parameters:
    x, y, z (list): Acceleration data
    time (float): Start time
    labels, starts, ends (list): Classification results from acceleration_data function
    """
    # Create synthetic time array assuming fixed sampling rate
    time_array = np.linspace(time, time + 0.01 * len(x), len(x))
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot acceleration data
    ax1.plot(time_array, x, 'r-', label='X acceleration')
    ax1.plot(time_array, y, 'g-', label='Y acceleration')
    ax1.plot(time_array, z, 'b-', label='Z acceleration')
    
    # Calculate and plot magnitude
    mag = np.sqrt(np.array(x)**2 + np.array(y)**2 + np.array(z)**2)
    ax1.plot(time_array, mag, 'k-', label='Magnitude', alpha=0.5)
    
    ax1.set_title('Acceleration Data')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot activity classifications
    colors = {'stair': 'red', 'no_stairs': 'green'}
    
    for i in range(len(labels)):
        label = labels[i]
        start = starts[i]
        end = ends[i]
        
        ax2.barh(y=0, width=end-start, left=start, height=0.6, 
                 color=colors.get(label, 'gray'), alpha=0.7, 
                 edgecolor='black', linewidth=0.5)
        
        # Add text label if segment is wide enough
        if end - start > 0.2:
            ax2.text(start + (end-start)/2, 0, label, 
                     ha='center', va='center', fontsize=10)
    
    ax2.set_yticks([])
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Activity Classification')
    
    # Ensure the x-axis limits match between plots
    ax1.set_xlim(time_array[0], time_array[-1])
    ax2.set_xlim(time_array[0], time_array[-1])
    
    plt.tight_layout()
    plt.savefig('activity_classification.png', dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Activity Classification from Acceleration Data')
    parser.add_argument('--model', default='best_model.keras', help='Path to the trained model file')
    parser.add_argument('--data', help='Path to CSV file with acceleration data (optional)')
    parser.add_argument('--visualize', action='store_true', help='Visualize the results')
    
    args = parser.parse_args()
    
    if args.data:
        # Load data from CSV file
        try:
            data = pd.read_csv(args.data)
            print(f"Loaded data from {args.data}")
            
            # Check for required columns
            required_cols = ['Time (s)', 'Linear Acceleration x (m/s^2)', 
                            'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)']
            
            if not all(col in data.columns for col in required_cols):
                print(f"CSV must contain columns: {required_cols}")
                return
            
            # Extract data
            time_col = data['Time (s)'].values
            x = data['Linear Acceleration x (m/s^2)'].values
            y = data['Linear Acceleration y (m/s^2)'].values
            z = data['Linear Acceleration z (m/s^2)'].values
            
            start_time = time_col[0]
            
            # Process the data
            labels, starts, ends = acceleration_data(x, y, z, start_time, args.model)
            
            # Display results
            print("\nClassification Results:")
            for i, (label, start, end) in enumerate(zip(labels, starts, ends)):
                print(f"Segment {i+1}: {start:.2f}s to {end:.2f}s - {label}")
            
            # Visualize if requested
            if args.visualize:
                visualize_results(x, y, z, start_time, labels, starts, ends)
                
        except Exception as e:
            print(f"Error processing data file: {e}")
    else:
        # Example usage with synthetic data
        print("No data file provided. Generating synthetic data for demonstration.")
        
        # Generate synthetic data
        t = np.linspace(0, 10, 500)  # 10 seconds of data
        x = np.sin(t) + np.random.normal(0, 0.1, len(t))
        y = np.cos(t) + np.random.normal(0, 0.1, len(t))
        z = np.sin(t*0.5) + np.random.normal(0, 0.1, len(t))
        
        print("Note: Using synthetic data. For real analysis, provide a CSV file with --data")
        print("Model inference will return random results without a proper trained model")
        
        # Process synthetic data
        labels, starts, ends = acceleration_data(x, y, z, 0.0, args.model)
        
        # Display results
        print("\nClassification Results:")
        for i, (label, start, end) in enumerate(zip(labels, starts, ends)):
            print(f"Segment {i+1}: {start:.2f}s to {end:.2f}s - {label}")
        
        # Visualize if requested
        if args.visualize:
            visualize_results(x, y, z, 0.0, labels, starts, ends)

if __name__ == "__main__":
    main()
