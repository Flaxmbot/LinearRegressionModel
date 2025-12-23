import axios from 'axios';
import { 
  UploadResponse, 
  TrainResponse, 
  PredictionRequest, 
  PredictionResponse, 
  TrainingParams,
  ModelsListResponse,
  ModelSelectionResponse,
  StatusResponse,
  TrainingStatusUpdate,
  VisualizationData,
  ModelStatus,
  DataProfile
} from '../types/api';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: '/api', // This will be proxied to localhost:8000
  timeout: 120000, // Increased timeout for training (2 minutes)
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Error:', {
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      message: error.message,
      code: error.code
    });
    
    // Handle specific error cases
    if (error.code === 'ECONNABORTED') {
      return Promise.reject(new Error('Request timeout - training may be taking longer than expected'));
    }
    
    if (error.response?.status === 0 || error.response?.status === undefined) {
      return Promise.reject(new Error('Network error - please check if the backend server is running on port 8000'));
    }
    
    if (error.response?.status === 403) {
      return Promise.reject(new Error('Access denied - CORS policy may be blocking the request'));
    }
    
    return Promise.reject(error);
  }
);

export class ApiService {
  /**
   * Upload and process CSV file
   */
  static async uploadFile(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await api.post<UploadResponse>('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to upload file');
    }
  }

  /**
   * Train the linear regression model
   */
  static async trainModel(params: TrainingParams, dataId: string): Promise<TrainResponse> {
    try {
      console.log('Starting model training with params:', params);
      
      const requestBody = {
        target: params.target,
        learning_rate: params.learningRate,
        epochs: params.epochs,
        model_name: params.modelName,
        data_id: dataId
      };

      console.log('Making training request to: /train');
      
      const response = await api.post<TrainResponse>('/train', requestBody);
      console.log('Training response received:', response.data);
      return response.data;
    } catch (error: any) {
      console.error('Training failed:', error);
      
      // Provide more specific error messages
      if (error.message.includes('timeout')) {
        throw new Error('Training is taking too long. Try reducing the number of epochs or check your data.');
      }
      
      if (error.message.includes('Network error')) {
        throw new Error('Cannot connect to backend server. Please ensure the server is running on port 8000.');
      }
      
      if (error.message.includes('CORS')) {
        throw new Error('CORS error - please refresh the page and try again.');
      }
      
      // Fall back to the original error message
      throw new Error(error.response?.data?.detail || error.message || 'Failed to train model');
    }
  }

  /**
   * Make predictions on new data
   */
  static async makePrediction(request: PredictionRequest): Promise<PredictionResponse> {
    try {
      const response = await api.post<PredictionResponse>('/predict', request);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to make prediction');
    }
  }

  /**
   * Get visualization data as blob (PNG image)
   */
  static async getVisualization(): Promise<Blob> {
    try {
      const response = await api.get('/visualize', {
        responseType: 'blob',
      });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get visualization');
    }
  }

  /**
   * Get list of all available models
   */
  static async getModels(): Promise<ModelsListResponse> {
    try {
      const response = await api.get<ModelsListResponse>('/models');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get models list');
    }
  }

  /**
   * Select a model by ID
   */
  static async selectModel(modelId: string): Promise<ModelSelectionResponse> {
    try {
      const response = await api.post<ModelSelectionResponse>(`/models/select/${modelId}`);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to select model');
    }
  }

  /**
   * Delete a model by ID
   */
  static async deleteModel(modelId: string): Promise<{message: string}> {
    try {
      const response = await api.delete(`/models/${modelId}`);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to delete model');
    }
  }

  /**
   * Get current status
   */
  static async getStatus(): Promise<StatusResponse> {
    try {
      const response = await api.get<StatusResponse>('/status');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get status');
    }
  }

  /**
   * Check if the backend is healthy
   */
  static async healthCheck(): Promise<boolean> {
    try {
      const response = await api.get('/', { timeout: 5000 }); // 5 second timeout for health check
      return response.status === 200;
    } catch (error) {
      console.warn('Health check failed:', error);
      return false;
    }
  }

  /**
   * Get training status for a specific model
   */
  static async getTrainingStatus(modelId: string): Promise<TrainingStatusUpdate> {
    try {
      const response = await api.get<TrainingStatusUpdate>(`/training/status/${modelId}`);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get training status');
    }
  }

  /**
   * Get all active training statuses
   */
  static async getAllTrainingStatuses(): Promise<TrainingStatusUpdate[]> {
    try {
      const response = await api.get<TrainingStatusUpdate[]>('/training/status');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get training statuses');
    }
  }

  /**
   * Get enhanced visualization data with multiple charts
   */
  static async getEnhancedVisualizationData(): Promise<VisualizationData> {
    try {
      const response = await api.get<VisualizationData>('/visualization/enhanced');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get enhanced visualization data');
    }
  }

  /**
   * Get specific visualization chart data
   */
  static async getVisualizationChart(chartType: string): Promise<any> {
    try {
      const response = await api.get(`/visualization/chart/${chartType}`);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || `Failed to get ${chartType} chart data`);
    }
  }

  /**
   * Get data profiling and analysis
   */
  static async getDataProfile(): Promise<DataProfile> {
    try {
      const response = await api.get<DataProfile>('/data/profile');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get data profile');
    }
  }

  /**
   * Get model performance insights
   */
  static async getModelInsights(): Promise<any> {
    try {
      const response = await api.get('/model/insights');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get model insights');
    }
  }

  /**
   * Get training recommendations
   */
  static async getTrainingRecommendations(): Promise<any> {
    try {
      const response = await api.get('/training/recommendations');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get training recommendations');
    }
  }

  /**
   * Export visualization as different formats
   */
  static async exportVisualization(format: 'png' | 'pdf' | 'svg' = 'png'): Promise<Blob> {
    try {
      const response = await api.get(`/visualization/export/${format}`, {
        responseType: 'blob',
      });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to export visualization');
    }
  }

  /**
   * Cancel ongoing training
   */
  static async cancelTraining(modelId: string): Promise<{message: string}> {
    try {
      const response = await api.post(`/training/cancel/${modelId}`);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to cancel training');
    }
  }

  /**
   * Get server metrics and health
   */
  static async getServerMetrics(): Promise<any> {
    try {
      const response = await api.get('/metrics');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get server metrics');
    }
  }

  /**
   * Download model as ZIP file
   */
  static async downloadModel(modelId: string): Promise<Blob> {
    try {
      const response = await api.get(`/models/download/${modelId}`, {
        responseType: 'blob',
      });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to download model');
    }
  }

  /**
   * Upload model from ZIP file
   */
  static async uploadModel(file: File, onProgress?: (progress: number) => void): Promise<{message: string, model_id: string}> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await api.post('/models/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            onProgress(progress);
          }
        },
      });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to upload model');
    }
  }

  /**
   * Save downloaded blob as file
   */
  static saveBlobAsFile(blob: Blob, filename: string): void {
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  }
}

export default ApiService;