import * as Location from 'expo-location';
import { Accelerometer } from 'expo-sensors';

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

const API_URL = 'http://localhost:8000'; // Update this with your backend URL

export const processSensorData = async (data: SensorData[]) => {
  try {
    const response = await fetch(`${API_URL}/process-sensor-data`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        recordingDate: new Date().toISOString(),
        sensorData: data,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result;
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
  cleanup();
  
  // Get the final data
  const recordedData = [...sensorData];
  
  // Clear the stored data
  sensorData.length = 0;
  
  return recordedData;
};

export const exportRecordingData = (data: SensorData[]) => {
  // Convert the data to a format suitable for export (e.g., JSON)
  const exportData = {
    recordingDate: new Date().toISOString(),
    sensorData: data,
  };

  return JSON.stringify(exportData, null, 2);
}; 