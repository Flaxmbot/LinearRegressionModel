import React, { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { BarChart3, TrendingUp, Play, RefreshCw, AlertCircle } from 'lucide-react';
import { PredictionRequest, PredictionResponse, TrainResponse } from '../types/api';
import ApiService from '../services/api';
import { InlineLoading, ButtonLoading } from './Loading';
import toast from 'react-hot-toast';

interface PredictionInterfaceProps {
  trainedModel: TrainResponse | null;
  isPredicting: boolean;
  setIsPredicting: (predicting: boolean) => void;
}

interface PredictionFormData {
  [key: string]: string;
}

const PredictionInterface: React.FC<PredictionInterfaceProps> = ({
  trainedModel,
  isPredicting,
  setIsPredicting
}) => {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [featureValues, setFeatureValues] = useState<{ [key: string]: number }>({});
  
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors }
  } = useForm<PredictionFormData>();

  // Reset form when model changes
  useEffect(() => {
    if (trainedModel) {
      reset();
      setPrediction(null);
      setFeatureValues({});
    }
  }, [trainedModel, reset]);

  const onSubmit = async (data: PredictionFormData) => {
    if (!trainedModel) {
      toast.error('Please train a model first');
      return;
    }

    // Convert string values to numbers
    const features = trainedModel.features_used.map(feature => {
      const value = parseFloat(data[feature] || '0');
      if (isNaN(value)) {
        throw new Error(`Invalid value for ${feature}`);
      }
      return value;
    });

    setIsPredicting(true);

    try {
      const request: PredictionRequest = { features };
      const response = await ApiService.makePrediction(request);
      setPrediction(response);
      toast.success('Prediction made successfully!');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Prediction failed';
      toast.error(message);
      setPrediction(null);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleInputChange = (feature: string, value: string) => {
    setFeatureValues(prev => ({
      ...prev,
      [feature]: parseFloat(value) || 0
    }));
  };

  const resetForm = () => {
    reset();
    setPrediction(null);
    setFeatureValues({});
  };

  if (!trainedModel) {
    return (
      <div className="card p-8 text-center">
        <AlertCircle className="h-12 w-12 text-slate-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-slate-700 mb-2">
          No Trained Model
        </h3>
        <p className="text-slate-500">
          Please train a model first before making predictions
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Model Info */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center space-x-2">
          <BarChart3 className="h-5 w-5 text-primary-600" />
          <span>Model Information</span>
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5 text-blue-600" />
              <span className="font-medium text-blue-800">Target Variable</span>
            </div>
            <p className="text-lg font-bold text-blue-900 mt-1">
              {trainedModel.target}
            </p>
          </div>
          
          <div className="bg-green-50 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5 text-green-600" />
              <span className="font-medium text-green-800">Features Used</span>
            </div>
            <p className="text-lg font-bold text-green-900 mt-1">
              {trainedModel.features_used.length}
            </p>
          </div>
        </div>

        <div className="mt-4">
          <h4 className="font-medium text-slate-700 mb-2">Features:</h4>
          <div className="flex flex-wrap gap-2">
            {trainedModel.features_used.map((feature, index) => (
              <span
                key={index}
                className="px-3 py-1 bg-slate-100 text-slate-700 rounded-full text-sm font-mono"
              >
                {feature}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Prediction Form */}
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">
            Enter Feature Values
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {trainedModel.features_used.map((feature, index) => (
              <div key={index}>
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  {feature}
                </label>
                <input
                  type="number"
                  step="any"
                  {...register(feature, {
                    required: `${feature} is required`,
                    valueAsNumber: true,
                    validate: (value) => !isNaN(value) || 'Must be a valid number'
                  })}
                  className={`input ${errors[feature] ? 'input-error' : ''}`}
                  placeholder={`Enter ${feature} value`}
                  onChange={(e) => handleInputChange(feature, e.target.value)}
                  disabled={isPredicting}
                />
                {errors[feature] && (
                  <p className="text-danger-600 text-sm mt-1">
                    {errors[feature]?.message}
                  </p>
                )}
              </div>
            ))}
          </div>

          {/* Action Buttons */}
          <div className="mt-6 pt-6 border-t border-slate-200 flex flex-col sm:flex-row gap-3">
            <button
              type="submit"
              disabled={isPredicting}
              className="btn-primary flex items-center justify-center space-x-2 px-6 py-3"
            >
              {isPredicting ? (
                <>
                  <ButtonLoading />
                  <span>Making Prediction...</span>
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  <span>Make Prediction</span>
                </>
              )}
            </button>
            
            <button
              type="button"
              onClick={resetForm}
              disabled={isPredicting}
              className="btn-secondary flex items-center justify-center space-x-2 px-6 py-3"
            >
              <RefreshCw className="h-4 w-4" />
              <span>Reset Form</span>
            </button>
          </div>
        </div>
      </form>

      {/* Prediction Result */}
      {prediction && (
        <div className="card p-6 border-success-200 bg-success-50">
          <h3 className="text-lg font-semibold text-success-800 mb-4 flex items-center space-x-2">
            <TrendingUp className="h-5 w-5" />
            <span>Prediction Result</span>
          </h3>
          
          <div className="bg-white rounded-lg p-6 border border-success-200">
            <div className="text-center">
              <p className="text-sm text-slate-600 mb-2">
                Predicted {trainedModel.target}
              </p>
              <p className="text-4xl font-bold text-success-700 mb-2">
                {prediction.prediction[0].toFixed(4)}
              </p>
              <p className="text-sm text-slate-500">
                Based on the input features
              </p>
            </div>
          </div>

          {/* Feature Values Summary */}
          <div className="mt-4">
            <h4 className="font-medium text-success-800 mb-2">Input Values:</h4>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
              {trainedModel.features_used.map((feature, index) => (
                <div key={index} className="bg-white rounded p-3 border border-success-200">
                  <p className="text-xs text-slate-500 font-medium">{feature}</p>
                  <p className="text-sm font-bold text-slate-800">
                    {featureValues[feature]?.toFixed(4) || '0.0000'}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <AlertCircle className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
          <div>
            <h4 className="font-medium text-blue-800 mb-2">Prediction Guidelines</h4>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• Enter values for all features used in training</li>
              <li>• Use realistic values within your data range for better accuracy</li>
              <li>• The prediction will be based on the trained linear regression model</li>
              <li>• Results are most reliable for values similar to your training data</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionInterface;