// API Response types
export interface UploadResponse {
  message: string;
  rows: number;
  columns: string[];
  data_preview: Record<string, any>[];
  data_shape: [number, number];
  filename: string;
  data_id: string;
}

export interface TrainResponse {
  message: string;
  model_id: string;
  model_name: string;
  features_used: string[];
  target: string;
  training_info: {
    learning_rate: number;
    epochs: number;
    samples_trained: number;
    feature_count: number;
  };
}

export interface PredictionResponse {
  prediction: number[];
  features_used: string[];
  model_info: {
    model_id: string;
    target_column: string;
    feature_count: number;
  };
}

// Model Management types
export interface ModelInfo {
  model_id: string;
  created_at: string;
  last_used: string;
  has_model: boolean;
  has_data: boolean;
  feature_count: number;
  target_column: string;
  data_shape: [number, number] | null;
  model_metadata: {
    model_name?: string;
    training_date: string;
    learning_rate: number;
    epochs: number;
    target_column: string;
    feature_columns: string[];
    training_samples: number;
    feature_count: number;
    filename: string;
  };
}

export interface ModelsListResponse {
  models: ModelInfo[];
  count: number;
  current_model_id: string | null;
}

export interface ModelSelectionResponse {
  message: string;
  model_id: string;
  model_info: {
    target_column: string;
    feature_count: number;
    feature_columns: string[];
    is_trained: boolean;
    created_at: string;
    last_used: string;
  };
}

export interface ApiError {
  detail: string;
}

// Training parameters interface
export interface TrainingParams {
  target: string;
  learningRate: number;
  epochs: number;
  modelName?: string;
}

// Prediction request interface
export interface PredictionRequest {
  features: number[];
}

// Training Status Types
export type TrainingStatus = 'initializing' | 'preprocessing' | 'training' | 'validating' | 'saving' | 'completed' | 'failed';

export interface TrainingStatusUpdate {
  model_id: string;
  status: TrainingStatus;
  progress: number; // 0-100
  message: string;
  current_epoch?: number;
  total_epochs?: number;
  loss?: number;
  validation_loss?: number;
}

// Enhanced Visualization Data Types
export interface VisualizationData {
  actual_vs_predicted: {
    actual: number[];
    predicted: number[];
    r2_score: number;
    mse: number;
    rmse: number;
    mae: number;
  };
  residuals: {
    residuals: number[];
    predicted_values: number[];
    standardized_residuals: number[];
  };
  feature_importance: {
    feature_names: string[];
    importance_scores: number[];
    coefficients: number[];
  };
  data_distribution: {
    target_distribution: {
      mean: number;
      median: number;
      std: number;
      min: number;
      max: number;
      quartiles: number[];
    };
    feature_distributions: Record<string, {
      mean: number;
      std: number;
      min: number;
      max: number;
    }>;
  };
  correlation_matrix: {
    feature_names: string[];
    correlation_values: number[][];
  };
  performance_metrics: {
    r2_score: number;
    mse: number;
    rmse: number;
    mae: number;
    explained_variance: number;
    mape: number;
  };
  data_insights: {
    data_quality_score: number;
    missing_values: Record<string, number>;
    outliers: Record<string, number[]>;
    feature_suggestions: string[];
    model_recommendations: string[];
  };
}

// Model Status with Training Tracking
export interface ModelStatus {
  model_id: string;
  status: 'training' | 'completed' | 'failed' | 'idle';
  training_progress: number;
  current_epoch?: number;
  total_epochs?: number;
  start_time?: string;
  end_time?: string;
  estimated_completion?: string;
  error_message?: string;
}

// Enhanced Status response interface
export interface StatusResponse {
  available_models: number;
  current_model_id: string | null;
  has_current_model: boolean;
  current_model_info: {
    has_data: boolean;
    has_model: boolean;
    is_model_trained: boolean;
    feature_count: number;
    target_column: string;
    data_shape: [number, number] | null;
    model_metadata: Record<string, any>;
  } | null;
  all_models: ModelInfo[];
  training_status?: ModelStatus[];
  server_health: {
    status: 'healthy' | 'degraded' | 'unhealthy';
    uptime: number;
    memory_usage: number;
  };
}

// Data Analysis Types
export interface DataProfile {
  dataset_info: {
    total_rows: number;
    total_columns: number;
    numeric_columns: string[];
    categorical_columns: string[];
    date_columns: string[];
    target_column?: string;
  };
  data_quality: {
    completeness_score: number;
    consistency_score: number;
    uniqueness_score: number;
    validity_score: number;
    overall_score: number;
  };
  statistical_summary: {
    numeric_summary: Record<string, {
      count: number;
      mean: number;
      std: number;
      min: number;
      q1: number;
      median: number;
      q3: number;
      max: number;
      missing_count: number;
    }>;
    categorical_summary: Record<string, {
      unique_count: number;
      most_frequent: string;
      frequency: number;
      missing_count: number;
    }>;
  };
  correlations: {
    high_correlations: Array<{
      feature1: string;
      feature2: string;
      correlation: number;
    }>;
    correlation_matrix: number[][];
    feature_names: string[];
  };
  recommendations: {
    data_cleaning: string[];
    feature_engineering: string[];
    modeling: string[];
    preprocessing: string[];
  };
}