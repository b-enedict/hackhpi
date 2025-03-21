import { StyleSheet, View, Text, FlatList, TouchableOpacity } from 'react-native';
import { Clock, Navigation } from 'lucide-react-native';

const recentRoutes = [
  {
    id: '1',
    from: 'Current Location',
    to: 'Central Park',
    date: '2024-02-20',
    duration: '15 min',
  },
  {
    id: '2',
    from: 'Home',
    to: 'Grand Central Station',
    date: '2024-02-19',
    duration: '25 min',
  },
];

export default function RoutesScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Recent Routes</Text>
      <FlatList
        data={recentRoutes}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <TouchableOpacity style={styles.routeCard}>
            <View style={styles.routeInfo}>
              <Text style={styles.routeName}>{item.to}</Text>
              <Text style={styles.routeFrom}>From: {item.from}</Text>
              <View style={styles.routeDetails}>
                <View style={styles.detailItem}>
                  <Clock size={16} color="#8E8E93" />
                  <Text style={styles.detailText}>{item.duration}</Text>
                </View>
                <Text style={styles.dateText}>{item.date}</Text>
              </View>
            </View>
            <Navigation size={24} color="#007AFF" />
          </TouchableOpacity>
        )}
        style={styles.list}
      />
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
  list: {
    paddingHorizontal: 15,
  },
  routeCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 15,
    marginVertical: 8,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.2,
    shadowRadius: 1.41,
    elevation: 2,
  },
  routeInfo: {
    flex: 1,
  },
  routeName: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 4,
  },
  routeFrom: {
    fontSize: 14,
    color: '#8E8E93',
    marginBottom: 8,
  },
  routeDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  detailItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  detailText: {
    marginLeft: 4,
    color: '#8E8E93',
    fontSize: 14,
  },
  dateText: {
    color: '#8E8E93',
    fontSize: 14,
  },
});