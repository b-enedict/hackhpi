import { HelpRequest } from '@/types'; // Assuming HelpRequest type is defined in your types file
import { startRecording, stopRecording } from '@/utils/recordingService';
import { useNavigation } from '@react-navigation/native';
import * as Location from 'expo-location';
import { router } from 'expo-router';
import { Navigation2, Search } from 'lucide-react-native';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Alert,
  Image,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import MapView, { Marker, Polyline } from 'react-native-maps';
import { API_URL } from '../../config/env';
import { useUser } from '../UserContext';

const POLLING_INTERVAL = 2000; // 2 seconds for polling help requests

interface DetectionEvent {
  id: number;
  detection_type: string;
  label: string;
  latitude: number;
  longitude: number;
  timestamp: string;
}

interface MapComponentProps {
  location: Location.LocationObject;
  detectionEvents: DetectionEvent[];
  route: any;
}
function MapScreen() {
  const { userType } = useUser(); // Get the user type from context

  const [location, setLocation] = useState<Location.LocationObject | null>(
    null
  );
  const [helpRequest, setHelpRequest] = useState<HelpRequest | null>(null); // State for the help request
  const [stairsData, setStairsData] = useState<StairsData>({ features: [] });
  const [isRecording, setIsRecording] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const pollingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(
    null
  );

  const [destination, setDestination] = useState<any | null>(null);
  const [route, setRoute] = useState<any[] | null>(null);
  const [detectionEvents, setDetectionEvents] = useState<DetectionEvent[]>([]);
  const recordingRef = useRef<{
    cleanup: () => void;
    getData: () => any[];
  } | null>(null);

  console.log(API_URL);

  const shouldShowHelpRequest = helpRequest && userType === 'normalUser';

  const navigation = useNavigation();

  // Hardcoded help request function (for now)
  const fetchHelpRequest = useCallback(async () => {
    try {
      // Uncomment and replace with real API when backend is ready
      // const response = await fetch(`${API_URL}/help-requests/`);
      // if (!response.ok) {
      //   throw new Error('Error fetching help requests');
      // }
      // const helpRequestData = await response.json();

      // Hardcoded data for now
      const helpRequestData = {
        userLocation: { latitude: 52.520008, longitude: 13.404954 },
        helpLocation: { latitude: 52.5203, longitude: 13.4051 },
        description:
          'Help needed to bypass stairs at this location, 5 km away from you',
      };

      setHelpRequest(helpRequestData); // Set hardcoded request
    } catch (error) {
      console.error('Error fetching help requests:', error);
      setHelpRequest(null); // Set to null in case of error
    }
  }, []);

  // Function to fetch detection events
  const fetchDetectionEvents = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/detections/`);
      if (!response.ok) {
        throw new Error('Failed to fetch detection events');
      }
      const events: DetectionEvent[] = await response.json();

      setDetectionEvents(events);
    } catch (error) {
      console.error('Error fetching detection events:', error);
      // Keep existing events or set empty array
      setDetectionEvents((prevEvents) => prevEvents || []);
    }
  }, []);

  // Setup polling for detection events
  //   useEffect(() => {
  //     // Initial fetch
  //     fetchDetectionEvents();

  //     // Start polling if not recording
  //     if (!isRecording) {
  //       pollingIntervalRef.current = setInterval(
  //         fetchDetectionEvents,
  //         POLLING_INTERVAL
  //       );
  //     }

  //     // Cleanup function
  //     return () => {
  //       if (pollingIntervalRef.current) {
  //         clearInterval(pollingIntervalRef.current);
  //         pollingIntervalRef.current = null;
  //       }
  //     };
  //   }, [isRecording, fetchDetectionEvents]);

  const handleRecordingPress = async () => {
    if (isRecording && recordingRef.current) {
      try {
        // Update UI state immediately
        setIsRecording(false);

        // Stop recording and send data in the background
        await stopRecording(recordingRef.current.cleanup);

        // Restart polling after recording stops
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
        }
        pollingIntervalRef.current = setInterval(
          fetchDetectionEvents,
          POLLING_INTERVAL
        );
      } catch (error) {
        console.error('Error stopping recording:', error);
        // Don't revert the UI state if there's an error
      }
    } else {
      try {
        // Update UI state immediately
        setIsRecording(true);

        // Start recording in the background
        const recording = await startRecording();
        recordingRef.current = recording;

        // Stop polling while recording
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
      } catch (error) {
        console.error('Error starting recording:', error);
        // Revert UI state if there's an error
        setIsRecording(false);
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

  useEffect(() => {
    (async () => {
      try {
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== 'granted') {
          setErrorMsg('Permission to access location was denied');
          return;
        }

        const location = await Location.getCurrentPositionAsync({});
        setLocation(location);
      } catch (error) {
        setErrorMsg('Error getting location');
      }
    })();

    // Start polling help requests
    pollingIntervalRef.current = setInterval(
      fetchHelpRequest,
      POLLING_INTERVAL
    );

    // Cleanup polling
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, [fetchHelpRequest]);

  const handleAccept = () => {
    if (helpRequest) {
      Alert.alert('Help Accepted', 'You are on your way to help!');
      navigation.navigate('HelpDetails', { helpRequest });
    }
  };

  const handleDecline = () => {
    Alert.alert('Help Declined', 'You declined the request.');
    setHelpRequest(null);
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
            {/* Mark stairs on the map */}
            {detectionEvents.map((event) => (
              <Marker
                key={event.id}
                coordinate={{
                  latitude: event.latitude,
                  longitude: event.longitude,
                }}
                title={event.detection_type}
                description={`${event.detection_type} detected at ${new Date(
                  event.timestamp
                ).toLocaleString()}`}
              >
                <View style={styles.markerContainer}>
                  <Image
                    source={require('../../assets/images/stairsMarker.png')}
                    style={styles.markerImage}
                  />
                </View>
              </Marker>
            ))}
            {shouldShowHelpRequest && (
              <Marker
                coordinate={helpRequest.helpLocation}
                title="Help Location"
                pinColor="red"
              />
            )}
            {shouldShowHelpRequest && (
              <Polyline
                coordinates={[location.coords, helpRequest.helpLocation]}
                strokeColor="#007AFF"
                strokeWidth={3}
              />
            )}
          </MapView>
        </View>
      )}

      {/* Button to initiate search */}
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
        style={[styles.floatingButton, isRecording && styles.recordingButton]}
        onPress={handleRecordingPress}
      >
        <Text style={styles.floatingButtonText}>
          {isRecording ? 'Stop Recording' : 'Start Recording'}
        </Text>
      </TouchableOpacity>

      {/* Display help request options */}
      {shouldShowHelpRequest && (
        <View style={styles.helpRequestCard}>
          <Text>Help Request: {helpRequest.description}</Text>
          <Text>
            Location: {helpRequest.helpLocation.latitude},{' '}
            {helpRequest.helpLocation.longitude}
          </Text>

          <TouchableOpacity style={styles.acceptButton} onPress={handleAccept}>
            <Text style={styles.buttonText}>Accept Request</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.declineButton}
            onPress={handleDecline}
          >
            <Text style={styles.buttonText}>Decline Request</Text>
          </TouchableOpacity>
        </View>
      )}
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
    backgroundColor: 'transparent',
  },
  markerImage: {
    width: 27,
    height: 27,
    resizeMode: 'contain',
  },
  helpRequestCard: {
    position: 'absolute',
    bottom: 100,
    left: 20,
    right: 20,
    backgroundColor: '#f0f0f0',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  acceptButton: {
    backgroundColor: '#4CAF50',
    padding: 10,
    borderRadius: 5,
    marginTop: 10,
  },
  declineButton: {
    backgroundColor: '#FF5722',
    padding: 10,
    borderRadius: 5,
    marginTop: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
  },
});

export default MapScreen;
