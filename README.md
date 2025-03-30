# HackHPI - Stair Detection and Route Planning

A full-stack application that detects stairs using sensor data and provides accessible route planning.

## Components

### Backend (FastAPI)
- FastAPI server handling sensor data processing and route calculations
- PostgreSQL database for storing detection events
- AWS Location Service integration for route planning
- ML model for stair detection (placeholder implementation)

### Frontend (React Native/Expo)
- Mobile app for recording sensor data
- Map view showing detected stairs and calculated routes
- Real-time data visualization

## Prerequisites
- Python 3.8+
- Node.js 16+
- Docker and Docker Compose
- AWS Location Service API key

## Starting

1. Clone the repository:
```bash
git clone <repository-url>
cd hackhpi
```

2. Set up environment variables:
Create a `.env` file in the root directory:
```bash
AWS_LOCATION_SERVICE_API_KEY=your-api-key-here
```

3. Start the backend and database using Docker:
```bash
docker-compose up --build
```

4. Make the backend publicly available by using for example ngrok. Write the URL under which the backend is available in the `./frontend/.env` file as
```bash
EXPO_PUBLIC_API_URL=https://your-backend-url
```

5. Install frontend dependencies:
```bash
cd frontend
npm install
```

6. Start the frontend development server:
```bash
npm run dev
```

7. Download the Expo Go App on your smartphone, connect to the same network as the machine on which you are running the expo development server and scan the qr code, which appears upon execution of `npm run dev`

8. After loading the code you can use the app