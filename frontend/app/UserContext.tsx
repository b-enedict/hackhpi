// frontend/app/UserContext.tsx
import React, { createContext, useState, useEffect, useContext } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Text, View } from 'react-native'; // Import Text and View for loading screen

const UserContext = createContext(null);

export const UserProvider = ({ children }: { children: React.ReactNode }) => {
  const [userType, setUserType] = useState<string | null>(null);

  useEffect(() => {
    const fetchUserType = async () => {
      const storedUserType = await AsyncStorage.getItem('userType');
      setUserType(storedUserType); // Set userType from storage
    };

    fetchUserType();
  }, []);

  const saveUserType = async (type: string) => {
    await AsyncStorage.setItem('userType', type);
    setUserType(type); // Update userType in context
  };

  return (
    <UserContext.Provider value={{ userType, saveUserType }}>
      {children}
    </UserContext.Provider>
  );
};

export const useUser = () => useContext(UserContext);
