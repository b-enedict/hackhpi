import React from 'react';
import { View, Text, StyleSheet, Button } from 'react-native';
import { RouteProp, useRoute } from '@react-navigation/native';
import MapView, { Marker } from 'react-native-maps';
import { HelpRequest } from '@/types';

type HelpDetailsScreenRouteProp = RouteProp<
  { HelpDetails: { helpRequest: HelpRequest } },
  'HelpDetails'
>;

const HelpDetailsScreen: React.FC = () => {
  const route = useRoute<HelpDetailsScreenRouteProp>();
  const { helpRequest } = route.params;

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Help Request Details</Text>
      <Text>Description: {helpRequest.description}</Text>
      <Text>
        Location: {helpRequest.helpLocation.latitude},{' '}
        {helpRequest.helpLocation.longitude}
      </Text>

      <MapView
        style={styles.map}
        initialRegion={{
          latitude: helpRequest.helpLocation.latitude,
          longitude: helpRequest.helpLocation.longitude,
          latitudeDelta: 0.0922,
          longitudeDelta: 0.0421,
        }}
      >
        <Marker coordinate={helpRequest.helpLocation} title="Help Location" />
        <Marker
          coordinate={helpRequest.userLocation}
          title="Your Location"
          pinColor="blue"
        />
      </MapView>

      <Button
        title="Confirm Help Done"
        onPress={() => alert('Help confirmed!')}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  map: {
    width: '100%',
    height: 300,
    marginTop: 20,
  },
});

export default HelpDetailsScreen;
