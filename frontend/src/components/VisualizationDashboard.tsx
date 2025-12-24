import React, { useState, useEffect } from 'react';
import {
  BarChart3, Image, Download, RefreshCw, AlertCircle, Eye,
  TrendingUp, TrendingDown, Activity, PieChart, BarChart,
  Zap, Target, Layers, FileImage, Lightbulb, Brain,
  ChevronDown, ChevronUp, Filter, Search, Grid, List
} from 'lucide-react';
import { TrainResponse, VisualizationData } from '../types/api';
import ApiService from '../services/api';
import { InlineLoading, ButtonLoading } from './Loading';
import toast from 'react-hot-toast';

interface VisualizationDashboardProps {
  trainedModel: TrainResponse | null;
}

interface ChartCardProps {
  title: string;
  icon: React.ReactNode;
  description: string;
  isLoading: boolean;
  children: React.ReactNode;
  onExport?: () => void;
  onRefresh?: () => void;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
}

const ChartCard: React.FC<ChartCardProps> = ({
  title,
  icon,
  description,
  isLoading,
  children,
  onExport,
  onRefresh,
  isExpanded = true,
  onToggleExpand
}) => {
  return (
    <div className="card p-6 hover:shadow-lg transition-shadow duration-200">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-primary-100 rounded-lg">
            {icon}
          </div>
          <div>
            <h3 className="text-lg font-semibold text-slate-800">{title}</h3>
            <p className="text-sm text-slate-600">{description}</p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {onRefresh && (
            <button
              onClick={onRefresh}
              disabled={isLoading}
              className="p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
              title="Refresh chart"
            >
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
          )}

          {onExport && (
            <button
              onClick={onExport}
              className="p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
              title="Export chart"
            >
              <Download className="h-4 w-4" />
            </button>
          )}

          {onToggleExpand && (
            <button
              onClick={onToggleExpand}
              className="p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
              title={isExpanded ? "Collapse" : "Expand"}
            >
              {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>
          )}
        </div>
      </div>

      {isExpanded && (
        <div className="border border-slate-200 rounded-lg overflow-hidden bg-white min-h-[300px]">
          {isLoading ? (
            <div className="h-80 flex items-center justify-center">
              <InlineLoading message="Generating chart..." />
            </div>
          ) : (
            children
          )}
        </div>
      )}
    </div>
  );
};

const VisualizationDashboard: React.FC<VisualizationDashboardProps> = ({
  trainedModel
}) => {
  const [visualizationData, setVisualizationData] = useState<VisualizationData | null>(null);
  const [chartStates, setChartStates] = useState({
    actualVsPredicted: { isLoading: false, image: null as string | null },
    residuals: { isLoading: false, image: null as string | null },
    featureImportance: { isLoading: false, image: null as string | null },
    performance: { isLoading: false, image: null as string | null },
    distribution: { isLoading: false, image: null as string | null },
    correlation: { isLoading: false, image: null as string | null }
  });

  const [insights, setInsights] = useState<any[]>([]);
  const [dataProfile, setDataProfile] = useState<any>(null);
  const [modelInsights, setModelInsights] = useState<any[]>([]);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [expandedSections, setExpandedSections] = useState({
    charts: true,
    insights: true,
    profile: true,
    recommendations: true
  });

  const loadEnhancedVisualizationData = async () => {
    if (!trainedModel) return;

    // Show loading state for all charts
    setChartStates(prev => {
      const newState = { ...prev };
      Object.keys(newState).forEach(key => {
        newState[key as keyof typeof chartStates] = { ...newState[key as keyof typeof chartStates], isLoading: true };
      });
      return newState;
    });

    try {
      // Pass the model_id explicitly to ensure we get data for the correct model
      const modelId = trainedModel.model_id;
      console.log(`Generating visualization for model: ${modelId}`);

      const [vizData, dataProf, insights, modelIns, recs] = await Promise.all([
        ApiService.getEnhancedVisualizationData(modelId),
        ApiService.getDataProfile(modelId),
        ApiService.getModelInsights(modelId),
        ApiService.getModelInsights(modelId),
        ApiService.getTrainingRecommendations()
      ]);

      setVisualizationData(vizData);
      setDataProfile(dataProf);
      setInsights(insights);
      setModelInsights(modelIns);
      setRecommendations(recs);

      // Load individual charts after main data
      refreshAllCharts(modelId);

    } catch (error) {
      console.error('Error loading enhanced visualization data:', error);
      toast.error('Failed to load visualization data');

      // Reset loading states
      setChartStates(prev => {
        const newState = { ...prev };
        Object.keys(newState).forEach(key => {
          newState[key as keyof typeof chartStates] = { ...newState[key as keyof typeof chartStates], isLoading: false };
        });
        return newState;
      });
    }
  };

  const loadChart = async (chartType: keyof typeof chartStates, modelId?: string) => {
    if (!trainedModel) return;

    // Use provided modelId or fall back to trainedModel.model_id
    const targetModelId = modelId || trainedModel.model_id;

    setChartStates(prev => ({
      ...prev,
      [chartType]: { ...prev[chartType], isLoading: true }
    }));

    try {
      const blob = await ApiService.getVisualizationChart(chartType, targetModelId);
      const imageUrl = URL.createObjectURL(blob);

      setChartStates(prev => ({
        ...prev,
        [chartType]: { isLoading: false, image: imageUrl }
      }));
    } catch (error) {
      console.error(`Error loading ${chartType} chart:`, error);
      setChartStates(prev => ({
        ...prev,
        [chartType]: { isLoading: false, image: null }
      }));
      // Don't toast for every individual chart failure to avoid spamming
    }
  };

  const exportChart = async (chartType: string) => {
    try {
      const blob = await ApiService.getVisualizationChart(chartType);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${chartType}-chart-${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      toast.success(`${chartType} chart exported!`);
    } catch (error) {
      toast.error(`Failed to export ${chartType} chart`);
    }
  };

  const exportAllCharts = async () => {
    const chartTypes = Object.keys(chartStates) as (keyof typeof chartStates)[];

    for (const chartType of chartTypes) {
      if (chartStates[chartType].image) {
        await exportChart(chartType);
        await new Promise(resolve => setTimeout(resolve, 500)); // Small delay between exports
      }
    }
  };

  const refreshAllCharts = async (modelId?: string) => {
    const chartTypes = Object.keys(chartStates) as (keyof typeof chartStates)[];

    // Load charts sequentially to avoid overwhelming the server
    for (const chartType of chartTypes) {
      await loadChart(chartType, modelId);
    }
  };

  // Effect to reset state when model changes, but DO NOT auto-load
  useEffect(() => {
    if (trainedModel) {
      // Clear existing visualizations when model changes
      setVisualizationData(null);
      setDataProfile(null);
      setInsights([]);
      setModelInsights([]);
      setRecommendations([]);

      setChartStates({
        actualVsPredicted: { isLoading: false, image: null },
        residuals: { isLoading: false, image: null },
        featureImportance: { isLoading: false, image: null },
        performance: { isLoading: false, image: null },
        distribution: { isLoading: false, image: null },
        correlation: { isLoading: false, image: null }
      });
    }
  }, [trainedModel?.model_id]); // Only run when model ID changes

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  if (!trainedModel) {
    return (
      <div className="card p-8 text-center">
        <AlertCircle className="h-12 w-12 text-slate-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-slate-700 mb-2">
          No Trained Model
        </h3>
        <p className="text-slate-500">
          Please train a model first to generate visualizations and analysis
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-800">Enhanced Model Analytics</h2>
          <p className="text-slate-600 mt-1">
            Comprehensive analysis with multiple visualization perspectives
          </p>
        </div>

        <div className="flex space-x-3">
          <button
            onClick={() => loadEnhancedVisualizationData()}
            className="btn-primary flex items-center space-x-2 bg-blue-600 hover:bg-blue-700"
            disabled={Object.values(chartStates).some(s => s.isLoading)}
          >
            {Object.values(chartStates).some(s => s.isLoading) ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                <span>Generating...</span>
              </>
            ) : (
              <>
                <Activity className="h-4 w-4" />
                <span>Generate Analysis</span>
              </>
            )}
          </button>

          <button
            onClick={() => refreshAllCharts()}
            className="btn-secondary flex items-center space-x-2"
            disabled={!visualizationData || Object.values(chartStates).some(s => s.isLoading)}
          >
            <RefreshCw className="h-4 w-4" />
            <span>Refresh All</span>
          </button>

          <button
            onClick={exportAllCharts}
            className="btn-secondary flex items-center space-x-2"
            disabled={!visualizationData}
          >
            <Download className="h-4 w-4" />
            <span>Export All</span>
          </button>
        </div>
      </div>

      {/* Model Summary Card */}
      <div className="card p-6 bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-slate-800 flex items-center space-x-2">
            <Brain className="h-5 w-5 text-primary-600" />
            <span>Model Overview: {trainedModel.model_name || trainedModel.model_id}</span>
          </h3>

          <div className="flex items-center space-x-2">
            <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
              Ready to Visualize
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white/70 rounded-lg p-4 backdrop-blur-sm">
            <h4 className="font-medium text-slate-700">Target Variable</h4>
            <p className="text-lg font-bold text-slate-900 mt-1">
              {trainedModel.target}
            </p>
          </div>

          <div className="bg-white/70 rounded-lg p-4 backdrop-blur-sm">
            <h4 className="font-medium text-slate-700">Features</h4>
            <p className="text-lg font-bold text-slate-900 mt-1">
              {trainedModel.features_used.length}
            </p>
          </div>

          <div className="bg-white/70 rounded-lg p-4 backdrop-blur-sm">
            <h4 className="font-medium text-slate-700">Model Type</h4>
            <p className="text-lg font-bold text-slate-900 mt-1">
              Linear Regression
            </p>
          </div>

          <div className="bg-white/70 rounded-lg p-4 backdrop-blur-sm">
            <h4 className="font-medium text-slate-700">Status</h4>
            <p className="text-lg font-bold text-green-600 mt-1">
              {visualizationData ? 'Analyzed' : 'Pending Analysis'}
            </p>
          </div>
        </div>

        {!visualizationData && (
          <div className="mt-6 flex justify-center">
            <button
              onClick={() => loadEnhancedVisualizationData()}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium shadow-md hover:shadow-lg transition-all flex items-center space-x-2"
            >
              <Activity className="h-5 w-5" />
              <span>Generate Valid Visualizations for this Model</span>
            </button>
          </div>
        )}
      </div>

      {/* Data Insights Section */}
      {dataProfile && (
        <div className="card p-6">
          <div
            className="flex items-center justify-between cursor-pointer"
            onClick={() => toggleSection('profile')}
          >
            <h3 className="text-lg font-semibold text-slate-800 flex items-center space-x-2">
              <Activity className="h-5 w-5 text-blue-600" />
              <span>Data Profile & Insights</span>
            </h3>
            {expandedSections.profile ?
              <ChevronUp className="h-5 w-5 text-slate-500" /> :
              <ChevronDown className="h-5 w-5 text-slate-500" />
            }
          </div>

          {expandedSections.profile && (
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {dataProfile.summary && Object.entries(dataProfile.summary).map(([key, value]) => (
                <div key={key} className="bg-slate-50 rounded-lg p-4">
                  <h4 className="font-medium text-slate-700 capitalize">{key}</h4>
                  <p className="text-lg font-bold text-slate-900 mt-1">{value as string}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Charts Grid */}
      <div className="space-y-6">
        <div
          className="flex items-center justify-between cursor-pointer"
          onClick={() => toggleSection('charts')}
        >
          <h3 className="text-lg font-semibold text-slate-800 flex items-center space-x-2">
            <Grid className="h-5 w-5 text-green-600" />
            <span>Visualization Charts</span>
          </h3>
          {expandedSections.charts ?
            <ChevronUp className="h-5 w-5 text-slate-500" /> :
            <ChevronDown className="h-5 w-5 text-slate-500" />
          }
        </div>

        {expandedSections.charts && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Actual vs Predicted Chart */}
            <ChartCard
              title="Actual vs Predicted"
              icon={<Target className="h-5 w-5 text-blue-600" />}
              description="Scatter plot comparing actual values with model predictions"
              isLoading={chartStates.actualVsPredicted.isLoading}
              onExport={() => exportChart('actualVsPredicted')}
              onRefresh={() => loadChart('actualVsPredicted')}
            >
              {chartStates.actualVsPredicted.image ? (
                <img
                  src={chartStates.actualVsPredicted.image}
                  alt="Actual vs Predicted"
                  className="w-full h-auto max-h-80 object-contain"
                />
              ) : (
                <div className="h-80 flex flex-col items-center justify-center text-center p-8">
                  <Target className="h-12 w-12 text-slate-400 mb-4" />
                  <p className="text-slate-500">Click refresh to generate chart</p>
                </div>
              )}
            </ChartCard>

            {/* Residuals Analysis Chart */}
            <ChartCard
              title="Residuals Analysis"
              icon={<TrendingUp className="h-5 w-5 text-red-600" />}
              description="Distribution of residuals to check model assumptions"
              isLoading={chartStates.residuals.isLoading}
              onExport={() => exportChart('residuals')}
              onRefresh={() => loadChart('residuals')}
            >
              {chartStates.residuals.image ? (
                <img
                  src={chartStates.residuals.image}
                  alt="Residuals Analysis"
                  className="w-full h-auto max-h-80 object-contain"
                />
              ) : (
                <div className="h-80 flex flex-col items-center justify-center text-center p-8">
                  <TrendingUp className="h-12 w-12 text-slate-400 mb-4" />
                  <p className="text-slate-500">Click refresh to generate chart</p>
                </div>
              )}
            </ChartCard>

            {/* Feature Importance Chart */}
            <ChartCard
              title="Feature Importance"
              icon={<Layers className="h-5 w-5 text-green-600" />}
              description="Ranking of feature importance in the model"
              isLoading={chartStates.featureImportance.isLoading}
              onExport={() => exportChart('featureImportance')}
              onRefresh={() => loadChart('featureImportance')}
            >
              {chartStates.featureImportance.image ? (
                <img
                  src={chartStates.featureImportance.image}
                  alt="Feature Importance"
                  className="w-full h-auto max-h-80 object-contain"
                />
              ) : (
                <div className="h-80 flex flex-col items-center justify-center text-center p-8">
                  <Layers className="h-12 w-12 text-slate-400 mb-4" />
                  <p className="text-slate-500">Click refresh to generate chart</p>
                </div>
              )}
            </ChartCard>

            {/* Performance Metrics Chart */}
            <ChartCard
              title="Performance Metrics"
              icon={<Zap className="h-5 w-5 text-yellow-600" />}
              description="Model performance metrics and accuracy indicators"
              isLoading={chartStates.performance.isLoading}
              onExport={() => exportChart('performance')}
              onRefresh={() => loadChart('performance')}
            >
              {chartStates.performance.image ? (
                <img
                  src={chartStates.performance.image}
                  alt="Performance Metrics"
                  className="w-full h-auto max-h-80 object-contain"
                />
              ) : (
                <div className="h-80 flex flex-col items-center justify-center text-center p-8">
                  <Zap className="h-12 w-12 text-slate-400 mb-4" />
                  <p className="text-slate-500">Click refresh to generate chart</p>
                </div>
              )}
            </ChartCard>

            {/* Data Distribution Chart */}
            <ChartCard
              title="Data Distribution"
              icon={<BarChart className="h-5 w-5 text-purple-600" />}
              description="Distribution of target variable and features"
              isLoading={chartStates.distribution.isLoading}
              onExport={() => exportChart('distribution')}
              onRefresh={() => loadChart('distribution')}
            >
              {chartStates.distribution.image ? (
                <img
                  src={chartStates.distribution.image}
                  alt="Data Distribution"
                  className="w-full h-auto max-h-80 object-contain"
                />
              ) : (
                <div className="h-80 flex flex-col items-center justify-center text-center p-8">
                  <BarChart className="h-12 w-12 text-slate-400 mb-4" />
                  <p className="text-slate-500">Click refresh to generate chart</p>
                </div>
              )}
            </ChartCard>

            {/* Correlation Heatmap */}
            <ChartCard
              title="Correlation Heatmap"
              icon={<PieChart className="h-5 w-5 text-indigo-600" />}
              description="Correlation matrix between features"
              isLoading={chartStates.correlation.isLoading}
              onExport={() => exportChart('correlation')}
              onRefresh={() => loadChart('correlation')}
            >
              {chartStates.correlation.image ? (
                <img
                  src={chartStates.correlation.image}
                  alt="Correlation Heatmap"
                  className="w-full h-auto max-h-80 object-contain"
                />
              ) : (
                <div className="h-80 flex flex-col items-center justify-center text-center p-8">
                  <PieChart className="h-12 w-12 text-slate-400 mb-4" />
                  <p className="text-slate-500">Click refresh to generate chart</p>
                </div>
              )}
            </ChartCard>
          </div>
        )}
      </div>

      {/* Model Insights Section */}
      {insights && insights.length > 0 && (
        <div className="card p-6">
          <div
            className="flex items-center justify-between cursor-pointer"
            onClick={() => toggleSection('insights')}
          >
            <h3 className="text-lg font-semibold text-slate-800 flex items-center space-x-2">
              <Lightbulb className="h-5 w-5 text-yellow-600" />
              <span>AI/ML Insights & Analysis</span>
            </h3>
            {expandedSections.insights ?
              <ChevronUp className="h-5 w-5 text-slate-500" /> :
              <ChevronDown className="h-5 w-5 text-slate-500" />
            }
          </div>

          {expandedSections.insights && (
            <div className="mt-4 space-y-3">
              {insights.map((insight, index) => (
                <div key={index} className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="flex items-start space-x-3">
                    <Lightbulb className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                    <p className="text-blue-800">{insight}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Recommendations Section */}
      {recommendations && recommendations.length > 0 && (
        <div className="card p-6">
          <div
            className="flex items-center justify-between cursor-pointer"
            onClick={() => toggleSection('recommendations')}
          >
            <h3 className="text-lg font-semibold text-slate-800 flex items-center space-x-2">
              <Brain className="h-5 w-5 text-green-600" />
              <span>Model Improvement Recommendations</span>
            </h3>
            {expandedSections.recommendations ?
              <ChevronUp className="h-5 w-5 text-slate-500" /> :
              <ChevronDown className="h-5 w-5 text-slate-500" />
            }
          </div>

          {expandedSections.recommendations && (
            <div className="mt-4 space-y-3">
              {recommendations.map((recommendation, index) => (
                <div key={index} className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <div className="flex items-start space-x-3">
                    <Brain className="h-5 w-5 text-green-600 mt-0.5 flex-shrink-0" />
                    <p className="text-green-800">{recommendation}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Action Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card p-4 bg-gradient-to-r from-green-50 to-blue-50 border-green-200">
          <h4 className="font-semibold text-slate-800 mb-2 flex items-center space-x-2">
            <Target className="h-4 w-4 text-green-600" />
            <span>Next Steps</span>
          </h4>
          <ul className="text-sm text-slate-600 space-y-1">
            <li>• Use the prediction interface to test new values</li>
            <li>• Try different hyperparameters for better performance</li>
            <li>• Collect more diverse training data if needed</li>
            <li>• Export visualizations for reports and presentations</li>
          </ul>
        </div>

        <div className="card p-4 bg-gradient-to-r from-purple-50 to-pink-50 border-purple-200">
          <h4 className="font-semibold text-slate-800 mb-2 flex items-center space-x-2">
            <Zap className="h-4 w-4 text-purple-600" />
            <span>Performance Tips</span>
          </h4>
          <ul className="text-sm text-slate-600 space-y-1">
            <li>• Monitor residuals for patterns indicating model issues</li>
            <li>• Check feature importance for feature engineering opportunities</li>
            <li>• Use correlation analysis to identify redundant features</li>
            <li>• Consider ensemble methods for improved accuracy</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default VisualizationDashboard;