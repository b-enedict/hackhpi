import { useState } from 'react';
import { StyleSheet, View, TextInput, FlatList, Text, TouchableOpacity } from 'react-native';
import { router } from 'expo-router';
import { MapPin, Clock } from 'lucide-react-native';

const recentSearches = [
  { id: '1', name: 'Central Park', address: 'New York, NY' },
  { id: '2', name: 'Grand Central Station', address: '89 E 42nd St, New York' },
];

export default function SearchScreen() {
  const [searchQuery, setSearchQuery] = useState('');

  const handleSearch = (location) => {
    // In a real app, this would search for the location and navigate back to the map
    router.back();
  };

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.searchInput}
        placeholder="Search for a destination"
        value={searchQuery}
        onChangeText={setSearchQuery}
        autoFocus
      />

      <Text style={styles.sectionTitle}>Recent Searches</Text>
      <FlatList
        data={recentSearches}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <TouchableOpacity
            style={styles.searchItem}
            onPress={() => handleSearch(item)}>
            <View style={styles.iconContainer}>
              {item.id === '1' ? (
                <MapPin size={24} color="#007AFF" />
              ) : (
                <Clock size={24} color="#007AFF" />
              )}
            </View>
            <View style={styles.locationInfo}>
              <Text style={styles.locationName}>{item.name}</Text>
              <Text style={styles.locationAddress}>{item.address}</Text>
            </View>
          </TouchableOpacity>
        )}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    paddingTop: 60,
  },
  searchInput: {
    height: 50,
    backgroundColor: '#F2F2F7',
    marginHorizontal: 20,
    marginBottom: 20,
    borderRadius: 10,
    paddingHorizontal: 15,
    fontSize: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#8E8E93',
    marginLeft: 20,
    marginBottom: 10,
  },
  searchItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#F2F2F7',
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#F2F2F7',
    justifyContent: 'center',
    alignItems: 'center',
  },
  locationInfo: {
    marginLeft: 15,
  },
  locationName: {
    fontSize: 16,
    fontWeight: '500',
  },
  locationAddress: {
    fontSize: 14,
    color: '#8E8E93',
    marginTop: 2,
  },
});