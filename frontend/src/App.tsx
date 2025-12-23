import { useState, useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import ModelTraining from './components/ModelTraining';
import ModelSelector from './components/ModelSelector';
import PredictionInterface from './components/PredictionInterface';
import VisualizationDashboard from './components/VisualizationDashboard';
import { UploadResponse, TrainResponse, ModelInfo } from './types/api';
import ApiService from './services/api';

function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [uploadedData, setUploadedData] = useState<UploadResponse | null>(null);
  const [trainedModel, setTrainedModel] = useState<TrainResponse | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [modelsRefreshTrigger, setModelsRefreshTrigger] = useState(0);


  // Load initial model state from backend
  useEffect(() => {
    const loadInitialModelState = async () => {
      try {
        const response = await ApiService.getModels();
        if (response.current_model_id && response.models.length > 0) {
          // Find the current model from the models list
          const currentModel = response.models.find(m => m.model_id === response.current_model_id);
          if (currentModel) {
            setSelectedModel(currentModel);
            // Create a mock TrainResponse from model info for compatibility
            const mockTrainResponse: TrainResponse = {
              message: 'Default model loaded successfully',
              model_id: currentModel.model_id,
              model_name: currentModel.model_metadata?.model_name || currentModel.model_id,
              features_used: currentModel.model_metadata?.feature_columns || [],
              target: currentModel.target_column,
              training_info: {
                learning_rate: currentModel.model_metadata?.learning_rate || 0.01,
                epochs: currentModel.model_metadata?.epochs || 1000,
                samples_trained: currentModel.model_metadata?.training_samples || 0,
                feature_count: currentModel.feature_count
              }
            };
            setTrainedModel(mockTrainResponse);
          }
        }
      } catch (error) {
        console.error('Failed to load initial model state:', error);
        // Don't show error toast as this is not critical for app startup
      }
    };

    loadInitialModelState();
  }, []);



  const handleUploadSuccess = (response: UploadResponse) => {
    setUploadedData(response);
    setActiveTab('train'); // Auto-switch to training after upload
  };

  const handleTrainingSuccess = (response: TrainResponse) => {
    setTrainedModel(response);
    setModelsRefreshTrigger(prev => prev + 1); // Trigger model list refresh
    setActiveTab('models'); // Auto-switch to models after training
  };

  const handleModelSelected = (modelInfo: ModelInfo) => {
    setSelectedModel(modelInfo);
    // Create a mock TrainResponse from model info for compatibility
    const mockTrainResponse: TrainResponse = {
      message: 'Model selected successfully',
      model_id: modelInfo.model_id,
      model_name: modelInfo.model_metadata?.model_name || modelInfo.model_id,
      features_used: modelInfo.model_metadata?.feature_columns || [],
      target: modelInfo.target_column,
      training_info: {
        learning_rate: modelInfo.model_metadata?.learning_rate || 0.01,
        epochs: modelInfo.model_metadata?.epochs || 1000,
        samples_trained: modelInfo.model_metadata?.training_samples || 0,
        feature_count: modelInfo.feature_count
      }
    };
    setTrainedModel(mockTrainResponse);
    setActiveTab('predict'); // Auto-switch to predictions after model selection
  };

  const handleModelsRefresh = () => {
    setModelsRefreshTrigger(prev => prev + 1);
  };

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    // Auto-refresh models when switching to models tab
    if (tab === 'models') {
      setModelsRefreshTrigger(prev => prev + 1);
    }
  };



  const renderContent = () => {
    switch (activeTab) {
      case 'upload':
        return (
          <FileUpload
            onUploadSuccess={handleUploadSuccess}
            isUploading={isUploading}
            setIsUploading={setIsUploading}
          />
        );

      case 'train':
        return (
          <ModelTraining
            uploadedData={uploadedData}
            onTrainingSuccess={handleTrainingSuccess}
            isTraining={isTraining}
            setIsTraining={setIsTraining}
            onModelsRefresh={handleModelsRefresh}
          />
        );

      case 'models':
        return (
          <ModelSelector
            onModelSelected={handleModelSelected}
            refreshTrigger={modelsRefreshTrigger}
          />
        );

      case 'predict':
        return (
          <PredictionInterface
            trainedModel={trainedModel}
            isPredicting={isPredicting}
            setIsPredicting={setIsPredicting}
          />
        );

      case 'visualize':
        return (
          <VisualizationDashboard
            trainedModel={trainedModel}
          />
        );

      default:
        return null;
    }
  };

  const getTabTitle = () => {
    switch (activeTab) {
      case 'upload':
        return 'Upload Your Data';
      case 'train':
        return 'Train Your Model';
      case 'models':
        return 'Select Model';
      case 'predict':
        return 'Make Predictions';
      case 'visualize':
        return 'Visualize Results';
      default:
        return 'Linear Regression ML Dashboard';
    }
  };

  const getTabDescription = () => {
    switch (activeTab) {
      case 'upload':
        return 'Upload your CSV data file to get started with machine learning';
      case 'train':
        return 'Configure and train your linear regression model';
      case 'models':
        return 'Select from your trained models or train a new one';
      case 'predict':
        return 'Use your trained model to make predictions on new data';
      case 'visualize':
        return 'Visualize your model performance and results';
      default:
        return 'Complete machine learning pipeline for linear regression';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <Header activeTab={activeTab} onTabChange={handleTabChange} />

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Page Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gradient mb-2">
            {getTabTitle()}
          </h1>
          <p className="text-slate-600 max-w-2xl mx-auto">
            {getTabDescription()}
          </p>
        </div>

        {/* Progress Indicators */}
        <div className="flex items-center justify-center mb-8">
          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-2 px-4 py-2 rounded-full ${uploadedData ? 'bg-success-100 text-success-800' :
                activeTab === 'upload' ? 'bg-primary-100 text-primary-800' :
                  'bg-slate-100 text-slate-600'
              }`}>
              <div className={`w-2 h-2 rounded-full ${uploadedData ? 'bg-success-600' :
                  activeTab === 'upload' ? 'bg-primary-600' :
                    'bg-slate-400'
                }`}></div>
              <span className="text-sm font-medium">Upload</span>
            </div>

            <div className={`w-8 h-0.5 ${trainedModel ? 'bg-success-600' : 'bg-slate-300'
              }`}></div>

            <div className={`flex items-center space-x-2 px-4 py-2 rounded-full ${trainedModel ? 'bg-success-100 text-success-800' :
                activeTab === 'train' ? 'bg-primary-100 text-primary-800' :
                  'bg-slate-100 text-slate-600'
              }`}>
              <div className={`w-2 h-2 rounded-full ${trainedModel ? 'bg-success-600' :
                  activeTab === 'train' ? 'bg-primary-600' :
                    'bg-slate-400'
                }`}></div>
              <span className="text-sm font-medium">Train</span>
            </div>

            <div className={`w-8 h-0.5 ${selectedModel ? 'bg-success-600' : 'bg-slate-300'
              }`}></div>

            <div className={`flex items-center space-x-2 px-4 py-2 rounded-full ${selectedModel ? 'bg-success-100 text-success-800' :
                activeTab === 'models' ? 'bg-primary-100 text-primary-800' :
                  'bg-slate-100 text-slate-600'
              }`}>
              <div className={`w-2 h-2 rounded-full ${selectedModel ? 'bg-success-600' :
                  activeTab === 'models' ? 'bg-primary-600' :
                    'bg-slate-400'
                }`}></div>
              <span className="text-sm font-medium">Select</span>
            </div>

            <div className={`w-8 h-0.5 ${selectedModel ? 'bg-success-600' : 'bg-slate-300'
              }`}></div>

            <div className={`flex items-center space-x-2 px-4 py-2 rounded-full ${selectedModel ? 'bg-success-100 text-success-800' : 'bg-slate-100 text-slate-600'
              }`}>
              <div className={`w-2 h-2 rounded-full ${selectedModel ? 'bg-success-600' : 'bg-slate-400'
                }`}></div>
              <span className="text-sm font-medium">Predict</span>
            </div>
          </div>
        </div>

        {/* Tab Content */}
        <div className="fade-in">
          {renderContent()}
        </div>
      </main>

      {/* Toast Notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#fff',
            color: '#374151',
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
            border: '1px solid #e5e7eb',
            borderRadius: '0.75rem',
            padding: '12px 16px',
          },
          success: {
            iconTheme: {
              primary: '#10b981',
              secondary: '#fff',
            },
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#fff',
            },
          },
        }}
      />

      {/* Footer */}
      <footer className="bg-white/50 backdrop-blur-sm border-t border-slate-200/50 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center">
            <p className="text-slate-600 text-sm">
              Linear Regression ML Dashboard • Built with React, TypeScript & Tailwind CSS
            </p>
            <p className="text-slate-500 text-xs mt-1">
              A modern interface for machine learning model training and prediction
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;