export interface LocationCoordinates {
  latitude: number;
  longitude: number;
}

export interface HelpRequest {
  userLocation: LocationCoordinates;
  helpLocation: LocationCoordinates;
  description: string;
}
