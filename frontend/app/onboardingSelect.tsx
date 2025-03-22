// frontend/app/onboardingSelect.tsx
import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  SafeAreaView,
} from 'react-native';
import { useUser } from './UserContext';
import { router } from 'expo-router';

export default function OnboardingSelect() {
  const { saveUserType } = useUser(); // Get the save function from context

  const handleNormalUserSelect = () => {
    saveUserType('normalUser');
    router.replace('/(nonDisabledFlow)');
  };

  const handleDisabilityUserSelect = () => {
    saveUserType('disabilityUser');
    router.replace('/onboarding');
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Welcome to StairGuard</Text>
        <Text style={styles.subtitle}>
          Select your user type to get started
        </Text>
      </View>

      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={[styles.button, styles.disabilityUserButton]}
          onPress={handleDisabilityUserSelect}
        >
          <Text style={styles.buttonText}>User with Disabilities</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.button, styles.normalUserButton]}
          onPress={handleNormalUserSelect}
        >
          <Text style={styles.buttonText}>User without Disabilities</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#E5E5E5',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 24,
  },
  header: {
    marginBottom: 50,
    textAlign: 'center',
  },
  title: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#007AFF',
    marginBottom: 12,
  },
  subtitle: {
    fontSize: 18,
    color: '#666',
    fontWeight: '500',
  },
  buttonContainer: {
    width: '100%',
    marginTop: 40,
    alignItems: 'center', // Center the buttons
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    marginVertical: 12,
    borderRadius: 30,
    backgroundColor: '#007AFF',
    width: '80%', // Adjust the width to make buttons smaller
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 4,
  },
  normalUserButton: {
    backgroundColor: '#007AFF',
  },
  disabilityUserButton: {
    backgroundColor: '#00C851', // Green color for accessibility
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 10,
  },
});
