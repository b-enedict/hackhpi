import { StyleSheet, View, Text, Switch, TouchableOpacity } from 'react-native';
import { Bell, MapPin, Navigation, Shield } from 'lucide-react-native';
import { useState } from 'react';

export default function SettingsScreen() {
  const [notifications, setNotifications] = useState(true);
  const [locationTracking, setLocationTracking] = useState(true);
  const [alternativeRoutes, setAlternativeRoutes] = useState(true);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Settings</Text>
      
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Navigation Preferences</Text>
        
        <View style={styles.setting}>
          <View style={styles.settingInfo}>
            <Bell size={24} color="#007AFF" />
            <View style={styles.settingText}>
              <Text style={styles.settingTitle}>Notifications</Text>
              <Text style={styles.settingDescription}>
                Receive alerts about stairs and route changes
              </Text>
            </View>
          </View>
          <Switch
            value={notifications}
            onValueChange={setNotifications}
            trackColor={{ false: '#767577', true: '#81b0ff' }}
            thumbColor={notifications ? '#007AFF' : '#f4f3f4'}
          />
        </View>

        <View style={styles.setting}>
          <View style={styles.settingInfo}>
            <MapPin size={24} color="#007AFF" />
            <View style={styles.settingText}>
              <Text style={styles.settingTitle}>Location Tracking</Text>
              <Text style={styles.settingDescription}>
                Allow continuous location updates
              </Text>
            </View>
          </View>
          <Switch
            value={locationTracking}
            onValueChange={setLocationTracking}
            trackColor={{ false: '#767577', true: '#81b0ff' }}
            thumbColor={locationTracking ? '#007AFF' : '#f4f3f4'}
          />
        </View>

        <View style={styles.setting}>
          <View style={styles.settingInfo}>
            <Navigation size={24} color="#007AFF" />
            <View style={styles.settingText}>
              <Text style={styles.settingTitle}>Alternative Routes</Text>
              <Text style={styles.settingDescription}>
                Automatically suggest accessible routes
              </Text>
            </View>
          </View>
          <Switch
            value={alternativeRoutes}
            onValueChange={setAlternativeRoutes}
            trackColor={{ false: '#767577', true: '#81b0ff' }}
            thumbColor={alternativeRoutes ? '#007AFF' : '#f4f3f4'}
          />
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>About</Text>
        <TouchableOpacity style={styles.setting}>
          <View style={styles.settingInfo}>
            <Shield size={24} color="#007AFF" />
            <View style={styles.settingText}>
              <Text style={styles.settingTitle}>Privacy Policy</Text>
            </View>
          </View>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F2F2F7',
  },
  title: {
    fontSize: 24,
    fontWeight: '600',
    padding: 20,
    backgroundColor: '#FFFFFF',
  },
  section: {
    marginTop: 20,
    backgroundColor: '#FFFFFF',
    paddingVertical: 10,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#8E8E93',
    marginLeft: 20,
    marginBottom: 10,
  },
  setting: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
    paddingHorizontal: 20,
    backgroundColor: '#FFFFFF',
  },
  settingInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  settingText: {
    marginLeft: 15,
    flex: 1,
  },
  settingTitle: {
    fontSize: 16,
    fontWeight: '500',
  },
  settingDescription: {
    fontSize: 14,
    color: '#8E8E93',
    marginTop: 2,
  },
});