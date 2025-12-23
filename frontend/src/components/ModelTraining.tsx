import React, { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { Settings, Play, CheckCircle, AlertCircle, Target, Zap, ArrowRight, Activity, RefreshCw } from 'lucide-react';
import { TrainResponse, UploadResponse, TrainingParams, TrainingStatusUpdate } from '../types/api';
import ApiService from '../services/api';
import { InlineLoading, ButtonLoading } from './Loading';
import toast from 'react-hot-toast';

interface ModelTrainingProps {
  uploadedData: UploadResponse | null;
  onTrainingSuccess: (response: TrainResponse) => void;
  isTraining: boolean;
  setIsTraining: (training: boolean) => void;
  onModelsRefresh: () => void;
  onRedirectToModels?: () => void;
  trainingModelId?: string;
}

const ModelTraining: React.FC<ModelTrainingProps> = ({
  uploadedData,
  onTrainingSuccess,
  isTraining,
  setIsTraining,
  onModelsRefresh,
  onRedirectToModels,
  trainingModelId
}) => {
  const [trainingProgress, setTrainingProgress] = useState<string>('');
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatusUpdate | null>(null);
  const [currentEpoch, setCurrentEpoch] = useState<number>(0);
  const [totalEpochs, setTotalEpochs] = useState<number>(0);
  const [currentTrainingModelId, setCurrentTrainingModelId] = useState<string | null>(null);
  const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState<string>('');
  const [showRedirectMessage, setShowRedirectMessage] = useState<boolean>(false);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  // Manual refresh function for training status - only triggered by user action
  const fetchTrainingStatus = async () => {
    if (!currentTrainingModelId) {
      return;
    }
    
    try {
      const response = await ApiService.getTrainingStatus(currentTrainingModelId);
      if (response) {
        setTrainingStatus(response);
        setTrainingProgress(response.status || 'Training in progress...');
        setCurrentEpoch(response.current_epoch || 0);
        setTotalEpochs(response.total_epochs || 0);
        
        // Sync isTraining state with actual backend status
        const isTrainingActive = response.status !== 'completed' && response.status !== 'failed';
        setIsTraining(isTrainingActive);
        
        // Redirect to models page when training completes
        if (response.status === 'completed') {
          setShowRedirectMessage(true);
          setTimeout(() => {
            onRedirectToModels?.();
          }, 2000);
        } else if (response.status === 'failed') {
          toast.error('Training failed. Please try again.');
        }
      }
    } catch (error) {
      console.error('Error fetching training status:', error);
    }
  };

  // Use provided trainingModelId or fallback to currentTrainingModelId
  const effectiveTrainingModelId = trainingModelId || currentTrainingModelId;
  
  const {
    register,
    handleSubmit,
    formState: { errors },
    watch
  } = useForm<TrainingParams>({
    defaultValues: {
      learningRate: 0.01,
      epochs: 1000,
      modelName: ''
    }
  });

  const watchedLearningRate = watch('learningRate');
  const watchedEpochs = watch('epochs');
  const watchedModelName = watch('modelName');

  const onSubmit = async (data: TrainingParams) => {
    if (!uploadedData) {
      toast.error('Please upload data first');
      return;
    }

    if (!data.target) {
      toast.error('Please select a target column');
      return;
    }

    setIsTraining(true);
    setTrainingProgress('Initializing training...');

    try {
      // Check if backend is accessible
      setTrainingProgress('Checking backend connection...');
      const isHealthy = await ApiService.healthCheck();
      if (!isHealthy) {
        throw new Error('Backend server is not responding. Please ensure the server is running on port 8000.');
      }
      
      setTrainingProgress('Training model...');
      const response = await ApiService.trainModel(data, uploadedData.data_id);
      
      // Extract and store the model_id for status polling
      if (response.model_id) {
        setCurrentTrainingModelId(response.model_id);
        console.log('Training started with model_id:', response.model_id);
      }
      
      toast.success('Model training started successfully!');
      onTrainingSuccess(response);
      setTrainingProgress('Training in progress...');
      
      // Trigger refresh of models list
      if (onModelsRefresh) {
        onModelsRefresh();
      }
    } catch (error) {
      console.error('Training error:', error);
      const message = error instanceof Error ? error.message : 'Training failed';
      toast.error(message);
      setTrainingProgress('');
    } finally {
      setIsTraining(false);
    }
  };

  // Manual refresh function for training status
  const refreshTrainingStatus = async () => {
    if (!currentTrainingModelId) {
      toast.error('No training in progress');
      return;
    }

    setIsRefreshing(true);
    try {
      await fetchTrainingStatus();
    } catch (error) {
      console.error('Error refreshing training status:', error);
      toast.error('Failed to refresh training status');
    } finally {
      setIsRefreshing(false);
    }
  };

  if (!uploadedData) {
    return (
      <div className="card p-8 text-center">
        <AlertCircle className="h-12 w-12 text-slate-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-slate-700 mb-2">
          No Data Available
        </h3>
        <p className="text-slate-500">
          Please upload a CSV file first to train a model
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Data Summary */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-4">Data Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <Target className="h-5 w-5 text-blue-600" />
              <span className="font-medium text-blue-800">Total Rows</span>
            </div>
            <p className="text-2xl font-bold text-blue-900 mt-1">
              {uploadedData.rows.toLocaleString()}
            </p>
          </div>
          
          <div className="bg-green-50 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <Settings className="h-5 w-5 text-green-600" />
              <span className="font-medium text-green-800">Features</span>
            </div>
            <p className="text-2xl font-bold text-green-900 mt-1">
              {uploadedData.columns.length.toLocaleString()}
            </p>
          </div>
          
          <div className="bg-purple-50 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-5 w-5 text-purple-600" />
              <span className="font-medium text-purple-800">Data Cleaned</span>
            </div>
            <p className="text-sm text-purple-700 mt-1">
              Processed and ready for training
            </p>
          </div>
        </div>
      </div>

      {/* Available Columns */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-4">
          Available Columns
        </h3>
        <div className="bg-slate-50 rounded-lg p-4 max-h-40 overflow-y-auto custom-scrollbar">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {uploadedData.columns.map((column, index) => (
              <div
                key={index}
                className="bg-white px-3 py-2 rounded border text-sm font-mono text-slate-700"
              >
                {column}
              </div>
            ))}
          </div>
        </div>
        <p className="text-sm text-slate-500 mt-2">
          All numeric columns and encoded categorical columns are available for training
        </p>
      </div>

      {/* Training Configuration */}
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center space-x-2">
            <Zap className="h-5 w-5 text-primary-600" />
            <span>Training Configuration</span>
          </h3>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Target Column Selection */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Target Column *
              </label>
              <select
                {...register('target', { required: 'Please select a target column' })}
                className={`input ${errors.target ? 'input-error' : ''}`}
                disabled={isTraining}
              >
                <option value="">Select target column...</option>
                {uploadedData.columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
              {errors.target && (
                <p className="text-danger-600 text-sm mt-1">{errors.target.message}</p>
              )}
              <p className="text-sm text-slate-500 mt-1">
                This column will be predicted by the model
              </p>
            </div>

            {/* Model Name */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Model Name (Optional)
              </label>
              <input
                type="text"
                {...register('modelName')}
                placeholder="My Custom Model Name"
                className="input"
                disabled={isTraining}
              />
              <p className="text-sm text-slate-500 mt-1">
                Give your model a descriptive name (optional)
              </p>
            </div>

            {/* Learning Rate */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Learning Rate: {watchedLearningRate}
              </label>
              <input
                type="range"
                min="0.001"
                max="0.1"
                step="0.001"
                {...register('learningRate', { 
                  required: true,
                  valueAsNumber: true 
                })}
                className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                disabled={isTraining}
              />
              <div className="flex justify-between text-xs text-slate-500 mt-1">
                <span>0.001</span>
                <span>0.1</span>
              </div>
              <p className="text-sm text-slate-500 mt-1">
                Lower values = more precise but slower training
              </p>
            </div>

            {/* Epochs */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Epochs: {watchedEpochs}
              </label>
              <input
                type="range"
                min="100"
                max="5000"
                step="100"
                {...register('epochs', { 
                  required: true,
                  valueAsNumber: true 
                })}
                className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                disabled={isTraining}
              />
              <div className="flex justify-between text-xs text-slate-500 mt-1">
                <span>100</span>
                <span>5000</span>
              </div>
              <p className="text-sm text-slate-500 mt-1">
                More epochs = better accuracy but longer training time
              </p>
            </div>
          </div>

          {/* Training Button */}
          <div className="mt-6 pt-6 border-t border-slate-200">
            <button
              type="submit"
              disabled={isTraining}
              className="btn-primary w-full sm:w-auto flex items-center justify-center space-x-2 px-8 py-3"
            >
              {isTraining ? (
                <>
                  <ButtonLoading />
                  <span>Training in Progress...</span>
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  <span>Start Training</span>
                </>
              )}
            </button>
            
            {isTraining && (
              <div className="mt-4">
                <div className="flex items-center justify-between mb-2">
                  <InlineLoading message={trainingProgress} />
                  {currentTrainingModelId && (
                    <button
                      onClick={refreshTrainingStatus}
                      disabled={isRefreshing}
                      className="btn-secondary text-sm px-3 py-1 flex items-center space-x-1"
                      title="Refresh training status"
                    >
                      <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
                      <span>{isRefreshing ? 'Refreshing...' : 'Refresh'}</span>
                    </button>
                  )}
                </div>
                
                {/* Training Progress Details */}
                {trainingStatus && (
                  <div className="bg-slate-50 rounded-lg p-3 mt-2">
                    <div className="flex justify-between text-sm text-slate-600 mb-1">
                      <span>Status: {trainingStatus.status}</span>
                      <span>Progress: {trainingStatus.progress}%</span>
                    </div>
                    {trainingStatus.current_epoch && trainingStatus.total_epochs && (
                      <div className="text-sm text-slate-600">
                        Epoch: {trainingStatus.current_epoch} / {trainingStatus.total_epochs}
                      </div>
                    )}
                    <div className="w-full bg-slate-200 rounded-full h-2 mt-2">
                      <div 
                        className="bg-primary-600 h-2 rounded-full transition-all duration-300" 
                        style={{ width: `${trainingStatus.progress}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </form>

      {/* Training Tips */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <AlertCircle className="h-5 w-5 text-amber-600 mt-0.5 flex-shrink-0" />
          <div>
            <h4 className="font-medium text-amber-800 mb-2">Training Tips</h4>
            <ul className="text-sm text-amber-700 space-y-1">
              <li>• Start with default values and adjust based on results</li>
              <li>• Lower learning rates (0.001-0.01) are generally safer</li>
              <li>• More epochs improve accuracy but increase training time</li>
              <li>• The model will automatically handle data normalization</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelTraining;