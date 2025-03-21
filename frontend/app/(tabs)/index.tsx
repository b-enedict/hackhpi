import * as Location from 'expo-location';
import { router } from 'expo-router';
import { Navigation2, Search } from 'lucide-react-native';
import { useEffect, useState, useRef, useCallback } from 'react';
import {
  Alert,
  Platform,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  Image,
} from 'react-native';
import { startRecording, stopRecording } from '../../utils/recordingService';
import { API_URL } from '../../config/env';

interface DetectionEvent {
  id: number;
  detection_type: string;
  latitude: number;
  longitude: number;
  timestamp: string;
}

interface StairsFeature {
  geometry: {
    coordinates: [number, number];
    type: string;
  };
  properties: {
    type: string;
    description: string;
  };
}

interface StairsData {
  features: StairsFeature[];
}

interface MapComponentProps {
  location: Location.LocationObject;
  stairsData: StairsData;
  route: any; // Update this type based on your route data structure
}

const POLLING_INTERVAL = 2000; // 2 seconds

// Web-compatible map component
function MapComponent({ location, stairsData, route }: MapComponentProps) {
  if (Platform.OS === 'web') {
    return (
      <iframe
        src={`https://www.openstreetmap.org/export/embed.html?bbox=${
          location?.coords.longitude - 0.01
        },${location?.coords.latitude - 0.01},${
          location?.coords.longitude + 0.01
        },${location?.coords.latitude + 0.01}&layer=mapnik&marker=${
          location?.coords.latitude
        },${location?.coords.longitude}`}
        style={{
          border: 'none',
          width: '100%',
          height: '100%',
        }}
      />
    );
  }

  // Import react-native-maps dynamically for native platforms
  const MapView = require('react-native-maps').default;
  const { Marker, Polyline } = require('react-native-maps');

  return (
    <MapView
      style={styles.map}
      initialRegion={{
        latitude: location.coords.latitude,
        longitude: location.coords.longitude,
        latitudeDelta: 0.0922,
        longitudeDelta: 0.0421,
      }}
      showsUserLocation={true}
      showsMyLocationButton={true}
    >
      {stairsData.features.map((feature, index) => (
        <Marker
          key={index}
          coordinate={{
            latitude: feature.geometry.coordinates[1],
            longitude: feature.geometry.coordinates[0],
          }}
          title={feature.properties.type}
          description={feature.properties.description}
        />
      ))}
      {route && (
        <Polyline coordinates={route} strokeColor="#007AFF" strokeWidth={3} />
      )}
    </MapView>
  );
}

export default function MapScreen() {
  const [location, setLocation] = useState<Location.LocationObject | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [destination, setDestination] = useState<any | null>(null);
  const [route, setRoute] = useState<any[] | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [stairsData, setStairsData] = useState<StairsData>({ features: [] });
  const recordingRef = useRef<{ cleanup: () => void; getData: () => any[] } | null>(null);
  const pollingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  console.log(API_URL);
  

  // Function to fetch detection events
  const fetchDetectionEvents = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/detection`);
      if (!response.ok) {
        throw new Error('Failed to fetch detection events');
      }
      const data = await response.json();
      setStairsData({ features: data });
    } catch (error) {
      console.error('Error fetching detection events:', error);
    }
  }, []);

  // Initial location setup
  useEffect(() => {
    (async () => {
      try {
        let { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== 'granted') {
          setErrorMsg('Permission to access location was denied');
          return;
        }

        let location = await Location.getCurrentPositionAsync({});
        setLocation(location);
      } catch (error) {
        console.error('Error getting location:', error);
        // Set a default location if we can't get the user's location
        setLocation({
          coords: {
            latitude: 52.520008, // Default to Berlin
            longitude: 13.404954,
            altitude: 0,
            accuracy: 0,
            altitudeAccuracy: 0,
            heading: 0,
            speed: 0,
          },
          timestamp: Date.now(),
        });
      }
    })();
  }, []);

  // Setup polling for detection events
  useEffect(() => {
    // Initial fetch
    fetchDetectionEvents();

    // Start polling if not recording
    if (!isRecording) {
      pollingIntervalRef.current = setInterval(fetchDetectionEvents, POLLING_INTERVAL);
    }

    // Cleanup function
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [isRecording, fetchDetectionEvents]);

  const handleRecordingPress = async () => {
    try {
      if (!isRecording) {
        // Stop polling when starting recording
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
        
        // Start recording
        const recording = await startRecording();
        recordingRef.current = recording;
        setIsRecording(true);
      } else {
        // Stop recording
        if (recordingRef.current) {
          const recordedData = await stopRecording(recordingRef.current.cleanup);
          recordingRef.current = null;
          
          // Fetch latest data and restart polling
          await fetchDetectionEvents().catch(console.error); // Don't let fetch errors block the UI
          pollingIntervalRef.current = setInterval(fetchDetectionEvents, POLLING_INTERVAL);
        }
        setIsRecording(false);
      }
    } catch (error) {
      console.error('Recording error:', error);
      Alert.alert('Recording Error', 'Failed to start/stop recording');
      setIsRecording(false);
      
      // Restart polling if recording fails
      if (!pollingIntervalRef.current) {
        pollingIntervalRef.current = setInterval(fetchDetectionEvents, POLLING_INTERVAL);
      }
    }
  };

  const checkForStairs = (currentRoute: any) => {
    Alert.alert(
      'Stairs Detected',
      'This route contains stairs. Would you like to find an alternative route?',
      [
        {
          text: 'Yes',
          onPress: () => findAlternativeRoute(),
        },
        {
          text: 'No',
          style: 'cancel',
        },
      ]
    );
  };

  const findAlternativeRoute = () => {
    Alert.alert(
      'Finding alternative route',
      'Calculating new accessible route...'
    );
  };

  // Only show error message for location permission denial
  if (errorMsg === 'Permission to access location was denied') {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>{errorMsg}</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {location && (
        <View style={styles.mapContainer}>
          <MapComponent
            location={location}
            stairsData={stairsData}
            route={route}
          />
        </View>
      )}

      <TouchableOpacity
        style={styles.searchButton}
        onPress={() => router.push('/search')}
      >
        <Search color="#007AFF" size={24} />
        <Text style={styles.searchButtonText}>Where to?</Text>
      </TouchableOpacity>

      {destination && (
        <TouchableOpacity
          style={styles.navigationButton}
          onPress={() => checkForStairs(route)}
        >
          <Navigation2 color="#FFFFFF" size={24} />
          <Text style={styles.navigationButtonText}>Start Navigation</Text>
        </TouchableOpacity>
      )}

      <TouchableOpacity
        style={[
          styles.floatingButton,
          isRecording && styles.recordingButton
        ]}
        onPress={handleRecordingPress}
      >
        <Text style={styles.floatingButtonText}>
          {isRecording ? 'Stop Recording' : 'Start Recording'}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  mapContainer: {
    flex: 1,
    overflow: 'hidden',
  },
  map: {
    flex: 1,
  },
  errorText: {
    fontSize: 16,
    color: 'red',
    textAlign: 'center',
    marginTop: 20,
  },
  searchButton: {
    position: 'absolute',
    top: 50,
    left: 20,
    right: 20,
    backgroundColor: '#FFFFFF',
    borderRadius: 25,
    padding: 15,
    flexDirection: 'row',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  searchButtonText: {
    marginLeft: 10,
    fontSize: 16,
    color: '#8E8E93',
  },
  navigationButton: {
    position: 'absolute',
    bottom: 30,
    left: 20,
    right: 20,
    backgroundColor: '#007AFF',
    borderRadius: 25,
    padding: 15,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  navigationButtonText: {
    marginLeft: 10,
    fontSize: 16,
    color: '#FFFFFF',
    fontWeight: '600',
  },
  floatingButton: {
    position: 'absolute',
    bottom: 30,
    right: 20,
    backgroundColor: '#FF3B30',
    borderRadius: 25,
    padding: 15,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  recordingButton: {
    backgroundColor: '#34C759',
  },
  floatingButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  markerContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  markerImage: {
    width: 40,
    height: 40,
  },
});
