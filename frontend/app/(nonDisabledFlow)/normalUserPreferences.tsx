import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  TouchableOpacity,
  Alert,
} from 'react-native';
import * as Location from 'expo-location';
import MapView, { Circle, Marker, PROVIDER_DEFAULT } from 'react-native-maps';
import { GooglePlacesAutocomplete } from 'react-native-google-places-autocomplete'; // For Google Places search

const GOOGLE_API_KEY = 'AIzaSyD1XlZ2CRJZREwguoMxhRUmD1PJPkdG9Xs'; // Replace with your actual Google API key

const NormalUserPreferences: React.FC = () => {
  const [location, setLocation] = useState<Location.LocationObject | null>(
    null
  );
  const [range, setRange] = useState<number>(5); // Default 5 km
  const [searchLocation, setSearchLocation] = useState<{
    latitude: number;
    longitude: number;
  } | null>(null); // For searched location
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Request location permission and get user location
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
        setErrorMsg('Error getting location');
      }
    })();
  }, []);

  const handleSavePreferences = () => {
    if (location) {
      Alert.alert(
        'Preferences Saved',
        `Your support area is set to ${range} km around your location.`
      );
      // In a real app, you would send this data to your backend or local storage
    } else {
      Alert.alert('Error', 'Location not available');
    }
  };
  const handleLocationSelect = (data: any, details = null) => {
    console.log('Selected data:', data); // Inspect the data structure
    if (details) {
      const { lat, lng } = details?.geometry?.location;
      setSearchLocation({
        latitude: lat,
        longitude: lng,
      });
    } else {
      console.log('Details not found.');
    }
  };

  if (errorMsg) {
    return (
      <View style={styles.container}>
        <Text>{errorMsg}</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Set Your Support Preferences</Text>

      <Text style={styles.label}>Range of Support (in km)</Text>
      <TextInput
        style={styles.input}
        value={String(range)}
        onChangeText={(text) => setRange(Number(text))}
        keyboardType="numeric"
        placeholder="Enter range in kilometers"
      />

      <GooglePlacesAutocomplete
        placeholder="Search for a location"
        onPress={(data, details = null) => handleLocationSelect(data, details)} // Handle location select
        query={{
          key: GOOGLE_API_KEY,
          language: 'en',
        }}
        fetchDetails={true}
        onFail={(error) => console.error('Google Places API Error: ', error)} // Add this to log any issues with the Places API
        styles={{
          textInputContainer: styles.textInputContainer,
          textInput: styles.textInput,
          predefinedPlacesDescription: styles.predefinedPlacesDescription,
        }}
      />

      <View style={styles.mapContainer}>
        {location && (
          <MapView
            style={styles.map}
            initialRegion={{
              latitude: location.coords.latitude,
              longitude: location.coords.longitude,
              latitudeDelta: 0.0922,
              longitudeDelta: 0.0421,
            }}
            provider={PROVIDER_DEFAULT}
          >
            <Circle
              center={{
                latitude: location.coords.latitude,
                longitude: location.coords.longitude,
              }}
              radius={range * 1000} // Radius in meters
              strokeColor="rgba(0, 122, 255, 0.5)"
              fillColor="rgba(0, 122, 255, 0.2)"
            />
            {searchLocation && (
              <Marker
                coordinate={{
                  latitude: searchLocation.latitude,
                  longitude: searchLocation.longitude,
                }}
                title="Searched Location"
              />
            )}
          </MapView>
        )}
      </View>

      <TouchableOpacity style={styles.button} onPress={handleSavePreferences}>
        <Text style={styles.buttonText}>Save Preferences</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  label: {
    fontSize: 16,
    marginBottom: 10,
  },
  input: {
    height: 40,
    borderColor: '#ccc',
    borderWidth: 1,
    borderRadius: 5,
    marginBottom: 20,
    paddingLeft: 10,
  },
  mapContainer: {
    flex: 1,
    marginBottom: 20,
  },
  map: {
    flex: 1,
    borderRadius: 10,
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
  },
  textInputContainer: {
    width: '100%',
    backgroundColor: '#fff',
  },
  textInput: {
    height: 40,
    borderColor: '#ccc',
    borderWidth: 1,
    borderRadius: 5,
    marginBottom: 20,
    paddingLeft: 10,
  },
  predefinedPlacesDescription: {
    color: '#1faadb',
  },
});

export default NormalUserPreferences;
