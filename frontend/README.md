# Linear Regression ML Dashboard - Frontend

A beautiful, modern React TypeScript frontend for the Linear Regression ML API backend. This dashboard provides a complete machine learning workflow interface with file upload, model training, predictions, and visualization capabilities.

## 🚀 Features

- **Modern UI/UX**: Clean, professional design with Tailwind CSS
- **Drag & Drop File Upload**: Intuitive CSV file upload with validation
- **Interactive Model Training**: Configure learning rate and epochs with real-time feedback
- **Prediction Interface**: Easy-to-use form for making predictions
- **Data Visualization**: Beautiful charts showing actual vs predicted values
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Real-time Feedback**: Loading states, notifications, and error handling
- **Type Safety**: Full TypeScript implementation for better development experience

## 🛠️ Technology Stack

- **React 18** - Modern React with hooks and functional components
- **TypeScript** - Type safety and better development experience
- **Tailwind CSS** - Utility-first CSS framework for styling
- **Vite** - Fast build tool and development server
- **React Query** - API state management and caching
- **React Hook Form** - Performant form handling
- **React Dropzone** - File drag & drop functionality
- **React Hot Toast** - Beautiful notification system
- **Lucide React** - Modern icon library
- **Axios** - HTTP client for API requests

## 📁 Project Structure

```
frontend/
├── public/                 # Static assets
├── src/
│   ├── components/        # React components
│   │   ├── Header.tsx     # Navigation header
│   │   ├── Loading.tsx    # Loading states
│   │   ├── FileUpload.tsx # File upload component
│   │   ├── ModelTraining.tsx # Training configuration
│   │   ├── PredictionInterface.tsx # Prediction form
│   │   └── VisualizationDashboard.tsx # Data visualization
│   ├── services/          # API services
│   │   └── api.ts         # API client and endpoints
│   ├── types/             # TypeScript types
│   │   └── api.ts         # API response types
│   ├── App.tsx            # Main application component
│   ├── main.tsx           # Application entry point
│   └── index.css          # Global styles
├── index.html             # HTML template
├── package.json           # Dependencies and scripts
├── tailwind.config.js     # Tailwind CSS configuration
├── vite.config.ts         # Vite configuration
└── tsconfig.json          # TypeScript configuration
```

## 🚀 Quick Start

### Prerequisites

- Node.js 16+ and npm/yarn
- Backend API running on localhost:8000

### Installation

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   # or
   yarn install
   ```

2. **Start the development server:**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

3. **Open your browser:**
   Navigate to `http://localhost:3000`

### Build for Production

```bash
npm run build
# or
yarn build
```

## 🔧 Configuration

### API Configuration

The frontend is configured to proxy API requests to the backend. Update `vite.config.ts` if your backend runs on a different port:

```typescript
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000', // Update this if needed
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, '')
    }
  }
}
```

### Tailwind CSS

Customize the design system in `tailwind.config.js`. The project includes:

- Custom color palette (primary, secondary, success, danger)
- Custom animations and transitions
- Responsive design utilities
- Custom component classes

## 📊 Usage Workflow

### 1. Data Upload
- Drag and drop or select a CSV file
- File validation ensures proper format and size
- Data preview shows uploaded content
- Automatic data cleaning and preprocessing

### 2. Model Training
- Select target column from uploaded data
- Configure learning rate (0.001 - 0.1)
- Set number of epochs (100 - 5000)
- Monitor training progress

### 3. Make Predictions
- Enter values for all features
- Get instant predictions from trained model
- View prediction results with confidence

### 4. Visualization
- Generate actual vs predicted value plots
- Download visualization images
- Analyze model performance

## 🎨 Design Features

### Color Scheme
- **Primary**: Blue tones (#3b82f6 to #1e3a8a)
- **Secondary**: Slate grays (#f8fafc to #0f172a)
- **Success**: Green tones (#22c55e to #14532d)
- **Danger**: Red tones (#ef4444 to #7f1d1d)

### Components
- **Cards**: Glassmorphism effect with subtle shadows
- **Buttons**: Multiple variants (primary, secondary, success, danger)
- **Forms**: Modern inputs with focus states and validation
- **Loading**: Skeleton screens and progress indicators
- **Notifications**: Toast system with success/error states

### Animations
- Fade-in effects for page transitions
- Smooth hover states and interactions
- Loading spinners and progress bars
- Slide-up animations for modals

## 🔗 API Integration

### Endpoints Used

The frontend integrates with these backend endpoints:

- **POST /upload** - Upload CSV files
- **GET /train** - Train linear regression model
- **POST /predict** - Make predictions
- **GET /visualize** - Generate visualization plots

### Error Handling

- Network error detection
- API response validation
- User-friendly error messages
- Retry mechanisms for failed requests

## 🧪 Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Code Style

- TypeScript for type safety
- ESLint for code quality
- Prettier for formatting
- Component-based architecture
- Custom hooks for logic reuse

## 🌟 Key Features

### User Experience
- **Progressive Disclosure**: Step-by-step workflow
- **Visual Feedback**: Loading states and progress indicators
- **Error Prevention**: Input validation and guidance
- **Accessibility**: Keyboard navigation and screen reader support

### Performance
- **Code Splitting**: Lazy loading of components
- **Image Optimization**: Efficient loading of visualizations
- **Caching**: React Query for API response caching
- **Bundle Optimization**: Vite for fast builds

### Responsive Design
- **Mobile First**: Optimized for mobile devices
- **Breakpoints**: Tailored layouts for different screen sizes
- **Touch Friendly**: Large touch targets and gestures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Check the browser console for errors
- Ensure the backend API is running
- Verify network connectivity
- Review the API documentation

---

**Built with ❤️ using React, TypeScript, and Tailwind CSS**