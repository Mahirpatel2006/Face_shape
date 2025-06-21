# Face Shape Analysis Application

This is a full-stack application that analyzes face shapes using AI. It consists of a Flask backend for face detection and analysis, and a Next.js frontend with two different result display components.

## Features

- **Face Shape Detection**: Uses MediaPipe and a trained Random Forest model to detect face shapes (Heart, Oval, Round, Square)
- **Dual Result Display**: 
  - **AnalysisCard**: Beautiful, animated display with personality insights and styling recommendations
  - **ResultSection**: Detailed facial measurements and technical data
- **Real-time Processing**: Processes uploaded images and displays results immediately
- **Modern UI**: Next.js frontend with beautiful animations and responsive design

## Project Structure

```
Face_Detection/
├── app.py                 # Flask backend with face analysis logic
├── app/                   # Next.js frontend
│   ├── page.tsx          # Main application page
│   └── api/              # API routes for frontend-backend communication
├── components/           # React components
│   ├── result-section.tsx # Detailed measurements display
│   ├── analysis-card.tsx  # Enhanced result display with personality insights
│   └── upload-section.tsx # File upload component
├── uploads/              # Directory for uploaded images
├── templates/            # Flask templates
└── requirements.txt      # Python dependencies
```

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Node.js Dependencies

```bash
npm install
# or
pnpm install
```

### 3. Run the Application

#### Start the Flask Backend
```bash
python app.py
```
The Flask server will run on `http://localhost:5000`

#### Start the Next.js Frontend
```bash
npm run dev
# or
pnpm dev
```
The Next.js app will run on `http://localhost:3000`

## How It Works

1. **Image Upload**: Users upload images through the Next.js frontend
2. **Backend Processing**: Flask backend processes images using MediaPipe face detection
3. **Face Shape Analysis**: The trained Random Forest model predicts face shape
4. **Dual Display**: Results are shown in two formats:
   - **AnalysisCard**: Enhanced display with personality traits, characteristics, and styling advice
   - **ResultSection**: Technical measurements and facial proportions
5. **Processed Images**: Shows original image with facial landmarks and face shape label

## API Endpoints

- `POST /analyze` - Analyzes a face image and returns face shape results with measurements
- `GET /uploads/<filename>` - Serves processed images
- `POST /upload` - Handles file uploads (Next.js API route)

## Face Shapes Supported

- **Heart**: Wider forehead, pointed chin, romantic silhouette
- **Oval**: Balanced proportions, most versatile for styling
- **Round**: Soft curves, full cheeks, warm appearance
- **Square**: Strong jawline, defined angles, commanding presence

## Components

### AnalysisCard
- Beautiful animated display
- Personality insights and characteristics
- Styling recommendations
- Career compatibility suggestions
- Confidence meter and visual effects

### ResultSection
- Detailed facial measurements
- Technical data (face length, cheekbone width, etc.)
- Processed image with landmarks
- Jaw curve ratio and proportions

## Technologies Used

- **Backend**: Flask, MediaPipe, OpenCV, scikit-learn
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **AI/ML**: Random Forest model for face shape classification
- **Computer Vision**: MediaPipe for facial landmark detection

## Testing

Run the test script to verify the backend is working:

```bash
python test_setup.py
```

## Notes

- Both result components display the same analysis data in different formats
- The AnalysisCard provides a more user-friendly, personality-focused experience
- The ResultSection provides detailed technical measurements for analysis
- All processed images are saved with landmarks drawn on them
- CORS is enabled for frontend-backend communication 