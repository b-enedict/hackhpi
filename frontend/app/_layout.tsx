// frontend/app/_layout.tsx
import React from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { router } from 'expo-router';
import { UserProvider } from './UserContext';

export default function Layout() {
  return (
    <UserProvider>
      <Stack screenOptions={{ headerShown: false }}>
        <Stack.Screen name="onboardingSelect" />
        <Stack.Screen name="(disabledFlow)" />
        <Stack.Screen name="(nonDisabledFlow)" />
      </Stack>
      <StatusBar style="auto" />
    </UserProvider>
  );
}
