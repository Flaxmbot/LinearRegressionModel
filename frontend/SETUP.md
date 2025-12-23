# Frontend Setup and Integration Guide

## Quick Start

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Start Development Server
```bash
npm run dev
```
This will start the frontend at `http://localhost:3000`

### 3. Ensure Backend is Running
Make sure your FastAPI backend is running at `http://localhost:8000`

## Integration with Backend

### API Endpoints Mapping
The frontend uses these backend endpoints:

- **POST /upload** → File upload with drag & drop
- **GET /train?target=X&learning_rate=Y&epochs=Z** → Model training
- **POST /predict** → Make predictions on new data
- **GET /visualize** → Generate actual vs predicted plots

### CORS Configuration
If you encounter CORS issues, update your FastAPI backend to include:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Proxy Configuration
The Vite dev server is configured to proxy `/api` requests to `http://localhost:8000`. If your backend runs on a different port, update `vite.config.ts`.

## Features Overview

### 1. Data Upload
- Drag & drop CSV file upload
- File validation (type, size)
- Data preview and processing status
- Automatic data cleaning

### 2. Model Training
- Interactive training configuration
- Learning rate slider (0.001 - 0.1)
- Epochs slider (100 - 5000)
- Target column selection
- Real-time training progress

### 3. Predictions
- Dynamic form generation based on trained features
- Input validation and error handling
- Instant prediction results
- Feature value summary

### 4. Visualization
- Actual vs predicted scatter plot
- Model performance summary
- Downloadable visualizations
- Performance interpretation guide

## Design System

### Colors
- **Primary**: Blue gradient (#3b82f6 → #1e3a8a)
- **Success**: Green (#22c55e)
- **Danger**: Red (#ef4444)
- **Neutral**: Slate grays

### Components
- Glassmorphism cards with subtle shadows
- Smooth animations and transitions
- Responsive design (mobile-first)
- Loading states and progress indicators
- Toast notifications

### Typography
- **Font**: Inter (Google Fonts)
- **Hierarchy**: Clear heading structure
- **Readability**: Optimized line heights and spacing

## Development

### Project Structure
```
frontend/src/
├── components/     # React components
├── services/       # API integration
├── types/          # TypeScript definitions
├── App.tsx         # Main application
└── main.tsx        # Entry point
```

### Key Technologies
- **React 18** with hooks
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **Vite** for fast development
- **React Hook Form** for forms
- **React Dropzone** for file uploads
- **React Hot Toast** for notifications

## Production Build

### Build for Production
```bash
npm run build
```

### Preview Production Build
```bash
npm run preview
```

### Deployment
The built files in the `dist/` folder can be deployed to any static hosting service:
- Vercel
- Netlify
- AWS S3 + CloudFront
- GitHub Pages

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Ensure backend CORS is configured
   - Check proxy settings in vite.config.ts

2. **API Connection Failed**
   - Verify backend is running on correct port
   - Check network connectivity
   - Review browser console for errors

3. **File Upload Issues**
   - Ensure CSV format is correct
   - Check file size limits (10MB max)
   - Verify backend /upload endpoint

4. **TypeScript Errors**
   - Install all dependencies: `npm install`
   - Check tsconfig.json configuration
   - Review import paths

### Browser Console
Open browser developer tools and check the Console tab for:
- Network request errors
- JavaScript runtime errors
- API response validation issues

## Performance Optimizations

- **Code Splitting**: Automatic with Vite
- **Image Optimization**: Efficient loading of visualizations
- **Caching**: React Query for API responses
- **Bundle Size**: Optimized dependencies

## Accessibility

- Keyboard navigation support
- Screen reader compatibility
- High contrast color ratios
- Focus indicators
- Semantic HTML structure

---

This frontend provides a complete, production-ready interface for your Linear Regression ML API with modern design patterns and excellent user experience.