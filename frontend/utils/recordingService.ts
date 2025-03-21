import * as Location from 'expo-location';
import { Accelerometer } from 'expo-sensors';
import { API_URL } from '../config/env';

interface SensorData {
  timestamp: number;
  linearAcceleration: {
    x: number;
    y: number;
    z: number;
  };
  location: LocationData | null;
}

interface LocationData {
  latitude: number;
  longitude: number;
  altitude: number | null;
  accuracy: number | null;
  speed: number | null;
}

const sensorData: SensorData[] = [];

// Configure the update intervals (in milliseconds)
const LOCATION_UPDATE_INTERVAL = 1000; // 1 second
const ACCELERATION_UPDATE_INTERVAL = 100; // 100ms (10 times per second)

export const processSensorData = async (data: any) => {
  try {
    // Filter out entries with null locations
    const validData = data.filter((entry: any) => entry.location !== null);
    
    const response = await fetch(`${API_URL}/process-sensor-data/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(validData),
    });

    if (!response.ok) {
      throw new Error('Failed to process sensor data');
    }

    return await response.json();
  } catch (error) {
    console.error('Error processing sensor data:', error);
    throw error;
  }
};

export const startRecording = async () => {
  try {
    // Request permissions
    const [locationStatus, accelerometerStatus] = await Promise.all([
      Location.requestForegroundPermissionsAsync(),
      Accelerometer.requestPermissionsAsync(),
    ]);

    if (locationStatus.status !== 'granted' || !accelerometerStatus.granted) {
      throw new Error('Permissions required for recording');
    }

    // Configure accelerometer
    Accelerometer.setUpdateInterval(ACCELERATION_UPDATE_INTERVAL);

    // Start recording data
    let lastLocation: LocationData | null = null;
    
    // Set up location tracking
    const locationSubscription = await Location.watchPositionAsync(
      {
        accuracy: Location.Accuracy.BestForNavigation,
        timeInterval: LOCATION_UPDATE_INTERVAL,
        distanceInterval: 1, // minimum distance (meters) between updates
      },
      (location) => {
        lastLocation = {
          latitude: location.coords.latitude,
          longitude: location.coords.longitude,
          altitude: location.coords.altitude,
          accuracy: location.coords.accuracy,
          speed: location.coords.speed,
        };
      }
    );
    
    // Subscribe to accelerometer updates
    const accelerometerSubscription = Accelerometer.addListener(
      accelerometerData => {
        const timestamp = Date.now();
        
        // Store the sensor data
        sensorData.push({
          timestamp,
          linearAcceleration: {
            x: accelerometerData.x,
            y: accelerometerData.y,
            z: accelerometerData.z,
          },
          location: lastLocation,
        });
      }
    );

    return {
      cleanup: () => {
        // Cleanup function to be called when stopping recording
        locationSubscription.remove();
        accelerometerSubscription.remove();
      },
      getData: () => [...sensorData], // Return a copy of the recorded data
    };
  } catch (error) {
    console.error('Error starting recording:', error);
    throw error;
  }
};

export const stopRecording = async (cleanup: () => void) => {
  try {
    // Call cleanup first to stop all subscriptions
    cleanup();
    
    // Get the final data
    const recordedData = [...sensorData];
    
    // Send the data to the backend
    await processSensorData(recordedData);
    
    // Clear the stored data
    sensorData.length = 0;
    
    return recordedData;
  } catch (error) {
    console.error('Error stopping recording:', error);
    // Still clear the data even if sending fails
    sensorData.length = 0;
    throw error;
  }
};

export const exportRecordingData = (data: SensorData[]) => {
  // Convert the data to a format suitable for export (e.g., JSON)
  const exportData = {
    recordingDate: new Date().toISOString(),
    sensorData: data,
  };

  return JSON.stringify(exportData, null, 2);
}; 