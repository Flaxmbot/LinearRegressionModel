import React, { useState, useEffect, useRef } from 'react';
import { Database, Trash2, CheckCircle, Clock, Target, Layers, Activity, Download, Upload, FileArchive, X, AlertCircle } from 'lucide-react';
import { ModelInfo, TrainingStatusUpdate } from '../types/api';
import ApiService from '../services/api';
import toast from 'react-hot-toast';

interface ModelSelectorProps {
  onModelSelected: (modelInfo: ModelInfo) => void;
  refreshTrigger?: number; // Use this to trigger refresh when models change
  onTrainingStart?: (modelId: string) => void; // Callback when training starts
  onTrainingComplete?: (modelId: string) => void; // Callback when training completes
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ 
  onModelSelected, 
  refreshTrigger,
  onTrainingStart,
  onTrainingComplete
}) => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [currentModelId, setCurrentModelId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [selectingId, setSelectingId] = useState<string | null>(null);
  const [trainingStatuses, setTrainingStatuses] = useState<{[modelId: string]: TrainingStatusUpdate}>({});
  const [refreshingStatuses, setRefreshingStatuses] = useState(false);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load training statuses for models
  const loadTrainingStatuses = async () => {
    try {
      const response = await ApiService.getAllTrainingStatuses();
      // Handle both possible response formats
      let statusesArray: TrainingStatusUpdate[] = [];
      if (Array.isArray(response)) {
        statusesArray = response;
      } else if (response && Array.isArray((response as any).training_statuses)) {
        statusesArray = (response as any).training_statuses;
      }
      
      // Convert array to object with modelId as key
      const statusMap: {[modelId: string]: TrainingStatusUpdate} = {};
      statusesArray.forEach((status: TrainingStatusUpdate) => {
        if (status.model_id) {
          statusMap[status.model_id] = status;
        }
      });
      setTrainingStatuses(statusMap);
    } catch (error) {
      console.error('Failed to load training statuses:', error);
      // Don't show error toast for training status as it's not critical
    }
  };

  // Manually refresh training statuses
  const refreshTrainingStatuses = async () => {
    setRefreshingStatuses(true);
    try {
      await loadTrainingStatuses();
    } finally {
      setRefreshingStatuses(false);
    }
  };

  const loadModels = async () => {
    setLoading(true);
    try {
      const response = await ApiService.getModels();
      setModels(response.models);
      setCurrentModelId(response.current_model_id);
      
      // Load training statuses after loading models
      await loadTrainingStatuses();
    } catch (error) {
      console.error('Failed to load models:', error);
      toast.error('Failed to load models');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadModels();
  }, [refreshTrigger]);

  const handleModelSelect = async (model: ModelInfo) => {
    if (isModelTraining(model.model_id)) {
      toast.error('Cannot select a model that is currently training');
      return;
    }

    setSelectingId(model.model_id);
    try {
      const response = await ApiService.selectModel(model.model_id);
      
      // Update local state immediately
      setCurrentModelId(model.model_id);
      
      // Call parent callback
      onModelSelected(model);
      
      // Show success message
      const modelName = model.model_metadata?.model_name || model.model_id;
      toast.success(`Model "${modelName}" selected successfully`);
      
      // Refresh models to get updated current_model_id from backend
      // This ensures consistency with backend state
      setTimeout(() => {
        loadModels();
      }, 500);
      
    } catch (error: any) {
      console.error('Failed to select model:', error);
      const errorMessage = error.message || 'Failed to select model';
      toast.error(`Error selecting model: ${errorMessage}`);
    } finally {
      setSelectingId(null);
    }
  };

  const handleDeleteModel = async (modelId: string) => {
    const modelToDelete = models.find(m => m.model_id === modelId);
    const modelName = modelToDelete?.model_metadata?.model_name || modelId;
    
    if (!window.confirm(`Are you sure you want to delete the model "${modelName}"? This action cannot be undone.`)) {
      return;
    }

    // Prevent deletion if model is currently selected
    if (currentModelId === modelId) {
      toast.error('Cannot delete the currently selected model. Please select a different model first.');
      return;
    }

    // Prevent deletion if model is currently training
    if (isModelTraining(modelId)) {
      toast.error('Cannot delete a model that is currently training. Please wait for training to complete.');
      return;
    }

    setDeletingId(modelId);
    try {
      await ApiService.deleteModel(modelId);
      
      // Update local state immediately
      setModels(models.filter(m => m.model_id !== modelId));
      if (currentModelId === modelId) {
        setCurrentModelId(null);
      }
      
      toast.success(`Model "${modelName}" deleted successfully`);
    } catch (error: any) {
      console.error('Failed to delete model:', error);
      const errorMessage = error.message || 'Failed to delete model';
      toast.error(`Error deleting model: ${errorMessage}`);
    } finally {
      setDeletingId(null);
    }
  };

  const handleDownloadModel = async (modelId: string) => {
    const modelToDownload = models.find(m => m.model_id === modelId);
    const modelName = modelToDownload?.model_metadata?.model_name || modelId;
    
    if (isModelTraining(modelId)) {
      toast.error('Cannot download a model that is currently training');
      return;
    }

    setDownloadingId(modelId);
    try {
      const blob = await ApiService.downloadModel(modelId);
      const filename = `${modelName || modelId}_${new Date().toISOString().split('T')[0]}.zip`;
      ApiService.saveBlobAsFile(blob, filename);
      
      toast.success(`Model "${modelName}" downloaded successfully`);
    } catch (error: any) {
      console.error('Failed to download model:', error);
      const errorMessage = error.message || 'Failed to download model';
      toast.error(`Error downloading model: ${errorMessage}`);
    } finally {
      setDownloadingId(null);
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Validate file type
      if (!file.name.toLowerCase().endsWith('.zip')) {
        toast.error('Please select a ZIP file');
        return;
      }
      
      // Validate file size (max 100MB)
      if (file.size > 100 * 1024 * 1024) {
        toast.error('File size must be less than 100MB');
        return;
      }
      
      handleUploadModel(file);
    }
  };

  const handleUploadModel = async (file: File) => {
    setUploading(true);
    setUploadProgress(0);
    
    try {
      const response = await ApiService.uploadModel(file, (progress) => {
        setUploadProgress(progress);
      });
      
      toast.success(`Model uploaded successfully: ${response.model_id}`);
      
      // Refresh models list to show the newly uploaded model
      await loadModels();
      
      // Close modal and reset
      setShowUploadModal(false);
      setUploadProgress(0);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error: any) {
      console.error('Failed to upload model:', error);
      const errorMessage = error.message || 'Failed to upload model';
      toast.error(`Error uploading model: ${errorMessage}`);
    } finally {
      setUploading(false);
    }
  };

  const openUploadModal = () => {
    setShowUploadModal(true);
  };

  const closeUploadModal = () => {
    if (!uploading) {
      setShowUploadModal(false);
      setUploadProgress(0);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const getStatusIcon = (status: string | undefined) => {
    switch (status) {
      case 'training':
        return <Activity className="h-4 w-4 text-blue-500 animate-pulse" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'pending':
        return <Clock className="h-4 w-4 text-yellow-500" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status: string | undefined) => {
    switch (status) {
      case 'training':
        return 'bg-blue-50 border-blue-200 text-blue-800';
      case 'completed':
        return 'bg-green-50 border-green-200 text-green-800';
      case 'failed':
        return 'bg-red-50 border-red-200 text-red-800';
      case 'pending':
        return 'bg-yellow-50 border-yellow-200 text-yellow-800';
      default:
        return 'bg-slate-50 border-slate-200 text-slate-800';
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const isModelTraining = (modelId: string) => {
    const status = trainingStatuses[modelId];
    return status?.status === 'training';
  };

  const getTrainingProgress = (modelId: string) => {
    const status = trainingStatuses[modelId];
    if (status && status.current_epoch && status.total_epochs) {
      return Math.round((status.current_epoch / status.total_epochs) * 100);
    }
    return 0;
  };

  if (loading) {
    return (
      <div className="card p-6">
        <div className="flex items-center space-x-3">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"></div>
          <span className="text-slate-600">Loading models...</span>
        </div>
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div className="card p-8 text-center">
        <Database className="h-12 w-12 text-slate-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-slate-700 mb-2">
          No Models Available
        </h3>
        <p className="text-slate-500">
          Train your first model to see it here. Models will appear automatically after training.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-slate-800 flex items-center space-x-2">
          <Database className="h-5 w-5" />
          <span>Available Models ({models.length})</span>
        </h3>
        <div className="flex items-center space-x-2">
          <button
            onClick={refreshTrainingStatuses}
            className="btn-secondary text-sm flex items-center space-x-1"
            disabled={refreshingStatuses}
            title="Refresh training status"
          >
            {refreshingStatuses ? (
              <>
                <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-current"></div>
                <span>Refreshing...</span>
              </>
            ) : (
              <>
                <Activity className="h-3 w-3" />
                <span>Refresh Status</span>
              </>
            )}
          </button>
          <button
            onClick={openUploadModal}
            className="btn-secondary text-sm flex items-center space-x-1"
            disabled={uploading}
            title="Upload model from ZIP file"
          >
            <Upload className="h-3 w-3" />
            <span>Upload Model</span>
          </button>
          <button
            onClick={loadModels}
            className="btn-secondary text-sm"
            disabled={loading}
          >
            Refresh Models
          </button>
        </div>
      </div>

      <div className="grid gap-4">
        {models.map((model) => (
          <div
            key={model.model_id}
            className={`card p-4 transition-all duration-200 hover:shadow-md ${
              currentModelId === model.model_id
                ? 'ring-2 ring-primary-500 bg-primary-50'
                : 'hover:bg-slate-50'
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-2">
                  <h4 className="font-semibold text-slate-800">
                    {model.model_metadata?.model_name || `Model ${model.model_id}`}
                  </h4>
                  {currentModelId === model.model_id && (
                    <>
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full font-medium">
                        Selected
                      </span>
                    </>
                  )}
                  {selectingId === model.model_id && (
                    <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full font-medium flex items-center space-x-1">
                      <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-current"></div>
                      <span>Selecting...</span>
                    </span>
                  )}
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm text-slate-600 mb-3">
                  <div className="flex items-center space-x-1">
                    <Target className="h-4 w-4" />
                    <span>Target: {model.target_column}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Layers className="h-4 w-4" />
                    <span>Features: {model.feature_count}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Clock className="h-4 w-4" />
                    <span>Created: {formatDate(model.created_at)}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Clock className="h-4 w-4" />
                    <span>Last used: {formatDate(model.last_used)}</span>
                  </div>
                </div>

                {/* Training Status */}
                {trainingStatuses[model.model_id] && (
                  <div className={`rounded-lg p-3 border ${getStatusColor(trainingStatuses[model.model_id].status)}`}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(trainingStatuses[model.model_id].status)}
                        <span className="font-medium capitalize">
                          {trainingStatuses[model.model_id].status || 'Unknown'}
                        </span>
                        {refreshingStatuses && (
                          <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-current"></div>
                        )}
                      </div>
                      {(trainingStatuses[model.model_id].status === 'training' || trainingStatuses[model.model_id].progress > 0) && (
                        <span className="text-sm font-mono">
                          {trainingStatuses[model.model_id].progress || getTrainingProgress(model.model_id)}%
                        </span>
                      )}
                    </div>
                    
                    {(trainingStatuses[model.model_id].status === 'training' || trainingStatuses[model.model_id].progress > 0) && (
                      <div className="w-full bg-white/30 rounded-full h-2 mb-2">
                        <div 
                          className="bg-current h-2 rounded-full transition-all duration-300"
                          style={{ width: `${trainingStatuses[model.model_id].progress || getTrainingProgress(model.model_id)}%` }}
                        />
                      </div>
                    )}
                    
                    {trainingStatuses[model.model_id].current_epoch && trainingStatuses[model.model_id].total_epochs && (
                      <div className="text-xs opacity-75">
                        Epoch {trainingStatuses[model.model_id].current_epoch} of {trainingStatuses[model.model_id].total_epochs}
                      </div>
                    )}
                    
                    {trainingStatuses[model.model_id].message && (
                      <div className="text-xs opacity-75 mt-1">
                        {trainingStatuses[model.model_id].message}
                      </div>
                    )}

                  </div>
                )}

                {model.model_metadata && (
                  <div className="mt-2 text-xs text-slate-500">
                    <span>Training: {model.model_metadata.epochs} epochs, LR: {model.model_metadata.learning_rate}</span>
                    {model.model_metadata.filename && (
                      <span> • Data: {model.model_metadata.filename}</span>
                    )}
                  </div>
                )}
              </div>

              <div className="flex items-center space-x-2 ml-4">
                <button
                  onClick={() => handleModelSelect(model)}
                  disabled={currentModelId === model.model_id || isModelTraining(model.model_id) || selectingId === model.model_id}
                  className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                    currentModelId === model.model_id
                      ? 'bg-green-100 text-green-700 cursor-not-allowed'
                      : isModelTraining(model.model_id)
                      ? 'bg-gray-100 text-gray-500 cursor-not-allowed'
                      : selectingId === model.model_id
                      ? 'bg-blue-100 text-blue-700 cursor-not-allowed'
                      : 'bg-primary-100 text-primary-700 hover:bg-primary-200'
                  }`}
                >
                  {currentModelId === model.model_id ? 'Selected' : 
                   isModelTraining(model.model_id) ? 'Training...' :
                   selectingId === model.model_id ? 'Selecting...' : 'Select'}
                  {selectingId === model.model_id && (
                    <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-current ml-1"></div>
                  )}
                </button>
                
                <button
                  onClick={() => handleDownloadModel(model.model_id)}
                  disabled={downloadingId === model.model_id || isModelTraining(model.model_id)}
                  className="p-1 text-blue-500 hover:text-blue-700 hover:bg-blue-50 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  title={isModelTraining(model.model_id) ? 'Cannot download a model that is training' : 'Download model as ZIP'}
                >
                  {downloadingId === model.model_id ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                  ) : (
                    <Download className="h-4 w-4" />
                  )}
                </button>
                
                <button
                  onClick={() => handleDeleteModel(model.model_id)}
                  disabled={deletingId === model.model_id || currentModelId === model.model_id || isModelTraining(model.model_id)}
                  className="p-1 text-red-500 hover:text-red-700 hover:bg-red-50 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  title={currentModelId === model.model_id ? 'Cannot delete the currently selected model' : 
                         isModelTraining(model.model_id) ? 'Cannot delete a model that is training' : 'Delete model'}
                >
                  {deletingId === model.model_id ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-red-500"></div>
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {currentModelId && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <span className="font-medium text-green-800">
              Model Selected: {models.find(m => m.model_id === currentModelId)?.model_metadata?.model_name || currentModelId}
            </span>
          </div>
          <div className="mt-2 text-sm text-green-700">
            <div className="flex items-center space-x-4">
              <span>Target: {models.find(m => m.model_id === currentModelId)?.target_column}</span>
              <span>Features: {models.find(m => m.model_id === currentModelId)?.feature_count}</span>
              <span>Status: {models.find(m => m.model_id === currentModelId)?.model_metadata?.model_name ? 'Ready' : 'Available'}</span>
            </div>
          </div>
          <p className="text-sm text-green-700 mt-1">
            You can now make predictions and view visualizations using this model.
          </p>
        </div>
      )}

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-slate-800 flex items-center space-x-2">
                <FileArchive className="h-5 w-5" />
                <span>Upload Model</span>
              </h3>
              <button
                onClick={closeUploadModal}
                disabled={uploading}
                className="text-slate-400 hover:text-slate-600 disabled:cursor-not-allowed"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div className="border-2 border-dashed border-slate-300 rounded-lg p-6 text-center">
                <Upload className="h-12 w-12 text-slate-400 mx-auto mb-4" />
                <h4 className="text-sm font-medium text-slate-700 mb-2">
                  Upload Model ZIP File
                </h4>
                <p className="text-xs text-slate-500 mb-4">
                  Select a ZIP file containing your trained model. Maximum file size: 100MB
                </p>
                
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".zip"
                  onChange={handleFileSelect}
                  disabled={uploading}
                  className="hidden"
                />
                
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploading}
                  className="btn-primary w-full flex items-center justify-center space-x-2"
                >
                  {uploading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      <span>Uploading...</span>
                    </>
                  ) : (
                    <>
                      <Upload className="h-4 w-4" />
                      <span>Choose ZIP File</span>
                    </>
                  )}
                </button>
              </div>
              
              {uploading && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-600">Uploading...</span>
                    <span className="text-slate-600">{uploadProgress}%</span>
                  </div>
                  <div className="w-full bg-slate-200 rounded-full h-2">
                    <div 
                      className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                </div>
              )}
              
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <h5 className="text-sm font-medium text-blue-800 mb-1">
                  Model Requirements:
                </h5>
                <ul className="text-xs text-blue-700 space-y-1">
                  <li>• Must be a valid ZIP file containing a trained model</li>
                  <li>• Compatible with the current model format</li>
                  <li>• Maximum file size: 100MB</li>
                  <li>• Model must be from a compatible version</li>
                </ul>
              </div>
            </div>
            
            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={closeUploadModal}
                disabled={uploading}
                className="btn-secondary disabled:cursor-not-allowed"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelSelector;