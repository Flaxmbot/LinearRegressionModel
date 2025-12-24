import fastapi, numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter threading issues
import matplotlib.pyplot as plt
import seaborn as sns
import io
import brain
import os
import json
import pickle
import zipfile
import tempfile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import uuid
import logging
import sys
from typing import Optional, List, Dict, Any
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyRegressor


app = FastAPI()

# Add root endpoint for health checks
@app.get("/")
async def root():
    return {"message": "ML Model API is running", "status": "healthy"}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3001",
        "http://localhost:5173",  # Vite default port
        "http://127.0.0.1:5173",
        "https://linear-regression-model-nine.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Enhanced model state management with support for multiple models.
    Each model gets a unique ID and can be loaded/selected independently.
    """
    
    def __init__(self):
        self.models = {}  # Dictionary to store multiple models
        self.current_model_id = None
        self.model_directory = "models"  # Directory to store individual model files
        self.temp_data = {}  # Temporary storage for uploaded data

        # Enhanced model status tracking
        self.model_status = {}  # Track status: 'training', 'ready', 'error'

        # Training status tracking
        self.last_training_info = None

        # Create models directory if it doesn't exist
        os.makedirs(self.model_directory, exist_ok=True)

        # Load existing models
        self.load_all_models()

        logger.info(f"ModelManager initialized with {len(self.models)} existing models")
    
    def store_temp_data(self, data_id: str, cleaned_df):
        """Store data temporarily for training"""
        self.temp_data[data_id] = cleaned_df
    
    def get_temp_data(self, data_id: str):
        """Get temporarily stored data"""
        return self.temp_data.get(data_id)
    
    def clear_temp_data(self, data_id: str):
        """Clear temporary data"""
        if data_id in self.temp_data:
            del self.temp_data[data_id]
    
    def generate_model_id(self) -> str:
        """Generate a unique model ID"""
        return f"model_{uuid.uuid4().hex[:8]}"
    
    def get_model_file_path(self, model_id: str) -> str:
        """Get the file path for a model"""
        return os.path.join(self.model_directory, f"{model_id}.pkl")
    
    def get_model_metadata_file_path(self, model_id: str) -> str:
        """Get the metadata file path for a model"""
        return os.path.join(self.model_directory, f"{model_id}_metadata.json")
    
    def validate_model(self, model_id: str) -> Dict[str, Any]:
        """Validate a model's integrity and return validation results"""
        validation_result = {
            'model_id': model_id,
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'model_exists': False,
            'metadata_exists': False,
            'model_loadable': False,
            'model_trained': False,
            'data_available': False
        }
        
        try:
            # Check if model files exist
            model_file = self.get_model_file_path(model_id)
            metadata_file = self.get_model_metadata_file_path(model_id)
            
            validation_result['model_exists'] = os.path.exists(model_file)
            validation_result['metadata_exists'] = os.path.exists(metadata_file)
            
            if not validation_result['model_exists']:
                validation_result['errors'].append('Model file not found')
                return validation_result
                
            if not validation_result['metadata_exists']:
                validation_result['warnings'].append('Metadata file not found')
            
            # Try to load model
            try:
                state_data = self.load_model(model_id)
                validation_result['model_loadable'] = True
                
                # Check if model is trained
                if state_data.get('trained_model') and hasattr(state_data['trained_model'], 'is_trained'):
                    validation_result['model_trained'] = state_data['trained_model'].is_trained
                    if not validation_result['model_trained']:
                        validation_result['warnings'].append('Model is not trained')
                else:
                    validation_result['warnings'].append('Trained model or training status not found')
                
                # Check data availability
                validation_result['data_available'] = state_data.get('cleaned_df') is not None
                
                validation_result['is_valid'] = True
                
            except Exception as e:
                validation_result['errors'].append(f'Model loading failed: {str(e)}')
                
        except Exception as e:
            validation_result['errors'].append(f'Validation error: {str(e)}')
            
        logger.info(f"Model validation for {model_id}: {'VALID' if validation_result['is_valid'] else 'INVALID'}")
        return validation_result
    
    def update_model_status(self, model_id: str, status: str):
        """Update model status (training, ready, error)"""
        valid_statuses = ['training', 'ready', 'error']
        if status not in valid_statuses:
            raise ValueError(f"Invalid status '{status}'. Must be one of: {valid_statuses}")
        
        self.model_status[model_id] = {
            'status': status,
            'updated_at': datetime.now().isoformat()
        }
        logger.info(f"Model {model_id} status updated to: {status}")
    
    def get_model_status(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get current model status"""
        return self.model_status.get(model_id)
    
    def save_model(self, model_id: str, cleaned_df, trained_model, feature_cols, target_col, model_metadata):
        """Save a model with its associated data"""
        try:
            # Set status to training during save process
            self.update_model_status(model_id, 'training')

            # Save only a sample of cleaned_df to reduce memory usage (max 1000 rows)
            sample_size = min(1000, len(cleaned_df))
            cleaned_df_sample = cleaned_df.sample(n=sample_size, random_state=42) if len(cleaned_df) > sample_size else cleaned_df

            # Save model state
            state_data = {
                'cleaned_df': cleaned_df_sample,
                'trained_model': trained_model,
                'feature_cols': feature_cols,
                'target_col': target_col,
                'model_metadata': model_metadata
            }
            
            with open(self.get_model_file_path(model_id), 'wb') as f:
                pickle.dump(state_data, f)
            
            # Save metadata as JSON
            metadata = {
                'model_id': model_id,
                'created_at': datetime.now().isoformat(),
                'last_used': datetime.now().isoformat(),
                'has_model': trained_model is not None,
                'has_data': cleaned_df_sample is not None,
                'feature_count': len(feature_cols),
                'target_column': target_col,
                'data_shape': cleaned_df_sample.shape if cleaned_df_sample is not None else None,
                'original_data_shape': cleaned_df.shape if cleaned_df is not None else None,
                'model_metadata': model_metadata
            }
            
            with open(self.get_model_metadata_file_path(model_id), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Add to models dictionary
            self.models[model_id] = metadata
            
            # Update status to ready if model is trained
            if trained_model and hasattr(trained_model, 'is_trained') and trained_model.is_trained:
                self.update_model_status(model_id, 'ready')
            else:
                self.update_model_status(model_id, 'error')
            
            logger.info(f"Model {model_id} saved successfully at {datetime.now().isoformat()}")
            
        except Exception as e:
            self.update_model_status(model_id, 'error')
            logger.error(f"Could not save model {model_id}: {e}")
            raise
    
    def load_model(self, model_id: str):
        """Load a specific model by ID"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            # Load model state
            with open(self.get_model_file_path(model_id), 'rb') as f:
                state_data = pickle.load(f)
            
            # Update current model
            self.current_model_id = model_id
            
            # Update last used timestamp
            self.models[model_id]['last_used'] = datetime.now().isoformat()
            with open(self.get_model_metadata_file_path(model_id), 'w') as f:
                json.dump(self.models[model_id], f, indent=2)
            
            print(f"Model {model_id} loaded successfully")
            return state_data
            
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            raise
    
    def load_all_models(self):
        """Load all existing models from disk"""
        try:
            if not os.path.exists(self.model_directory):
                return
            
            for filename in os.listdir(self.model_directory):
                if filename.endswith('_metadata.json'):
                    model_id = filename.replace('_metadata.json', '')
                    try:
                        with open(self.get_model_metadata_file_path(model_id), 'r') as f:
                            metadata = json.load(f)
                        self.models[model_id] = metadata
                        print(f"Found model {model_id} - {metadata.get('target_column', 'Unknown')} ({metadata.get('feature_count', 0)} features)")
                    except Exception as e:
                        print(f"Warning: Could not load metadata for {model_id}: {e}")
            
            print(f"Loaded {len(self.models)} existing models")
            
        except Exception as e:
            print(f"Warning: Could not load existing models: {e}")
    
    def list_models(self):
        """Get list of all available models"""
        return list(self.models.values())
    
    def delete_model(self, model_id: str):
        """Delete a model and its files"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            # Remove files
            for file_path in [self.get_model_file_path(model_id), self.get_model_metadata_file_path(model_id)]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Remove from models dictionary
            del self.models[model_id]
            
            # Clear current model if it was the deleted one
            if self.current_model_id == model_id:
                self.current_model_id = None
            
            print(f"Model {model_id} deleted successfully")
            
        except Exception as e:
            print(f"Error deleting model {model_id}: {e}")
            raise
    
    def get_current_model(self):
        """Get the currently selected model data"""
        if not self.current_model_id:
            return None
        
        if self.current_model_id not in self.models:
            self.current_model_id = None
            return None
        
        try:
            return self.load_model(self.current_model_id)
        except Exception as e:
            print(f"Warning: Could not load current model: {e}")
            self.current_model_id = None
            return None
    
    def get_status(self):
        """Get current status of the model manager"""
        current_model = self.get_current_model()
        
        return {
            'available_models': len(self.models),
            'current_model_id': self.current_model_id,
            'has_current_model': current_model is not None,
            'current_model_info': {
                'has_data': current_model['cleaned_df'] is not None if current_model else False,
                'has_model': current_model['trained_model'] is not None if current_model else False,
                'is_model_trained': getattr(current_model['trained_model'], 'is_trained', False) if current_model and current_model['trained_model'] else False,
                'feature_count': len(current_model['feature_cols']) if current_model else 0,
                'target_column': current_model['target_col'] if current_model else '',
                'data_shape': current_model['cleaned_df'].shape if (current_model is not None and current_model.get('cleaned_df') is not None) else None,
                'model_metadata': current_model['model_metadata'] if current_model else {}
            } if current_model else None,
            'all_models': self.list_models()
        }

# Global model manager instance
model_manager = ModelManager()

class PredictionInput(BaseModel):
    features: list[float]

@app.post("/predict")
async def make_prediction(input_data: PredictionInput):
    """Make prediction using the currently selected trained model with proper error handling."""
    
    # Get current model data
    current_model_data = model_manager.get_current_model()
    
    if not current_model_data:
        raise HTTPException(
            status_code=400, 
            detail="No model selected. Please select a trained model first."
        )
    
    trained_model = current_model_data['trained_model']
    feature_cols = current_model_data['feature_cols']
    
    # Check if model is trained
    if not hasattr(trained_model, 'is_trained') or not trained_model.is_trained:
        raise HTTPException(
            status_code=400, 
            detail="Selected model is not properly trained. Please retrain the model."
        )
    
    expected_features = len(feature_cols)
    if len(input_data.features) != expected_features:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected {expected_features} features, got {len(input_data.features)}. "
                   f"Available features: {feature_cols}"
        )
    
    try:
        input_array = np.array([input_data.features])
        prediction = trained_model.predict(input_array)

        print(f"Prediction successful: {prediction.flatten().tolist()}")

        return {
            "prediction": prediction.flatten().tolist(),
            "features_used": feature_cols,
            "model_info": {
                "model_id": model_manager.current_model_id,
                "target_column": current_model_data['target_col'],
                "feature_count": len(feature_cols)
            }
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

def comprehensive_data_cleaning(df):
    """
    Comprehensive data cleaning pipeline that ensures all data is numeric
    and compatible with numpy operations.
    """
    print(f"Starting comprehensive data cleaning - Original shape: {df.shape}")
    print(f"Original dtypes: {dict(df.dtypes)}")
    
    # Remove duplicates and rows with all NaN values
    df = df.drop_duplicates().dropna(how='all')
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    print(f"Boolean columns: {bool_cols}")
    
    # Handle missing values
    # For numeric columns: fill with mean
    for col in numeric_cols:
        if df[col].isna().any():
            mean_val = df[col].mean()
            if pd.isna(mean_val):  # All values are NaN
                df[col] = df[col].fillna(0)  # Fill with 0 if mean is NaN
            else:
                df[col] = df[col].fillna(mean_val)
    
    # For categorical columns: fill with mode or 'Unknown'
    for col in categorical_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
            else:
                df[col] = df[col].fillna("Unknown")
    
    # For boolean columns: fill with False
    for col in bool_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(False)
    
    # Remove outlier values from numeric columns
    for col in numeric_cols:
        if df[col].nunique() > 1:  # Only if there's variance
            try:
                q05 = df[col].quantile(0.05)
                q95 = df[col].quantile(0.95)
                df[col] = df[col].clip(lower=q05, upper=q95)
            except Exception as e:
                print(f"Warning: Could not clip column {col}: {e}")
    
    # Convert categorical columns to dummy variables
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"After pd.get_dummies(), new shape: {df.shape}")
    
    # Convert boolean columns to integers (0/1)
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)
    
    # CRITICAL: Ensure all columns are numeric
    # Convert any remaining non-numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # If conversion resulted in all NaN, fill with 0
                if df[col].isna().all():
                    df[col] = 0
                else:
                    df[col] = df[col].fillna(df[col].mean() if not df[col].mean() == 'nan' else 0)
            except Exception as e:
                print(f"Warning: Could not convert column {col} to numeric: {e}")
                # Drop problematic columns
                df = df.drop(columns=[col])
    
    # FINAL VALIDATION: Ensure all data is numeric
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        print(f"Warning: Non-numeric columns remaining: {non_numeric_cols}")
        # Force conversion one more time
        for col in non_numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except:
                df = df.drop(columns=[col])
    
    # Remove any infinite or NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    print(f"Final cleaned shape: {df.shape}")
    print(f"Final dtypes: {dict(df.dtypes)}")
    
    # Final safety check: convert to numeric array and back to ensure compatibility
    try:
        # Test that numpy operations work
        test_array = df.values
        np.std(test_array, axis=0)  # This should not fail
        print("SUCCESS: Data passed numpy compatibility test")
    except Exception as e:
        print(f"ERROR: Data failed numpy compatibility test: {e}")
        # Force everything to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Final conversion test
        test_array = df.values
        if test_array.dtype == object:
            raise ValueError("Data still has object dtype - cannot proceed")
    
    return df

@app.post("/upload")
async def process_upload(file: fastapi.UploadFile = fastapi.File(...)):
    """Upload and clean data with proper state management."""

    try:
        print(f"Processing file upload: {file.filename}")

        # Check file size limit (100MB)
        MAX_FILE_SIZE_MB = 100
        file_size = 0
        contents = b''
        chunk_size = 1024 * 1024  # 1MB chunks
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            contents += chunk
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB. Your file is {file_size / (1024*1024):.1f}MB."
                )

        file_size_mb = len(contents) / (1024 * 1024)
        logger.info(f"File uploaded: {file.filename}, size: {file_size_mb:.2f} MB")

        df = pd.read_csv(io.BytesIO(contents), low_memory=False)
        df_memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"DataFrame loaded - shape: {df.shape}, memory: {df_memory_mb:.2f} MB")

        print(f"Original data shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")

        # Apply comprehensive data cleaning
        cleaned_data = comprehensive_data_cleaning(df)
        cleaned_memory_mb = cleaned_data.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Data cleaned - shape: {cleaned_data.shape}, memory: {cleaned_memory_mb:.2f} MB")

        # Generate a data ID for temporary storage
        data_id = f"data_{uuid.uuid4().hex[:8]}"

        # Store in temporary storage
        model_manager.store_temp_data(data_id, cleaned_data)

        response_data = {
            "message": "Data cleaned and ready for training",
            "rows": len(cleaned_data),
            "columns": list(cleaned_data.columns),
            "data_preview": cleaned_data.head(5).to_dict(orient="records"),
            "data_shape": cleaned_data.shape,
            "filename": file.filename,
            "data_id": data_id
        }

        print(f"Data upload successful - Shape: {cleaned_data.shape}")
        print(f"   Data ID for training: {data_id}")
        return response_data

    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process upload: {str(e)}"
        )

@app.post("/train")
async def train_model_with_data(request: dict):
    """Train a new model with uploaded data and save it."""

    # Initialize training tracking
    training_status = 'failed'
    error_message = None
    start_time = None
    end_time = None

    try:
        # Extract parameters from request
        target = request.get('target')
        learning_rate = request.get('learning_rate', 0.01)
        epochs = request.get('epochs', 1000)
        model_name = request.get('model_name')
        data_id = request.get('data_id')
        
        if not target:
            raise HTTPException(status_code=400, detail="Target column is required")
        
        if not data_id:
            raise HTTPException(status_code=400, detail="Data ID is required")
        
        # Get the cleaned data from temporary storage
        cleaned_df = model_manager.get_temp_data(data_id)
        if cleaned_df is None:
            raise HTTPException(
                status_code=400,
                detail="Data not found or expired. Please upload your data again."
            )

        cleaned_memory_mb = cleaned_df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Retrieved cleaned data - shape: {cleaned_df.shape}, memory: {cleaned_memory_mb:.2f} MB")

        if target not in cleaned_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target}' not found in data. "
                       f"Available columns: {list(cleaned_df.columns)}"
            )

        print(f"Starting model training - Target: {target}, LR: {learning_rate}, Epochs: {epochs}")

        # Prepare training data
        y = cleaned_df[target].values
        X = cleaned_df.drop(columns=[target])
        feature_cols = list(X.columns)
        X_values = X.values

        x_memory_mb = sys.getsizeof(X_values) / (1024 * 1024)
        y_memory_mb = sys.getsizeof(y) / (1024 * 1024)
        logger.info(f"Training arrays prepared - X: {X_values.shape}, Y: {y.shape}, X memory: {x_memory_mb:.2f} MB, Y memory: {y_memory_mb:.2f} MB")

        print(f"Training data prepared - Features: {len(feature_cols)}, Samples: {X_values.shape[0]}")
        print(f"Target column: {target}")
        print(f"Feature columns: {feature_cols}")

        # Create and train model
        trained_model = brain.Brain(feature_size=X_values.shape[1], action_size=1)

        print(f"Training model with {X_values.shape[1]} features...")
        start_time = datetime.now()
        trained_model.train(X_values, y, learning_rate=learning_rate, epochs=epochs)
        end_time = datetime.now()
        training_status = 'completed'
        
        # Generate model ID and metadata
        model_id = model_manager.generate_model_id()
        if not model_name:
            model_name = f"Model_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_metadata = {
            'model_name': model_name,
            'training_date': datetime.now().isoformat(),
            'learning_rate': learning_rate,
            'epochs': epochs,
            'target_column': target,
            'feature_columns': feature_cols,
            'training_samples': X_values.shape[0],
            'feature_count': X_values.shape[1],
            'data_id': data_id,  # Store reference to original data
            'training_start': start_time.isoformat() if start_time else None,
            'training_end': end_time.isoformat() if end_time else None,
            'training_status': training_status,
            'error_message': error_message
        }
        
        # Save the model
        model_manager.save_model(
            model_id=model_id,
            cleaned_df=cleaned_df,
            trained_model=trained_model,
            feature_cols=feature_cols,
            target_col=target,
            model_metadata=model_metadata
        )

        # Set last training info
        model_manager.last_training_info = {
            'model_id': model_id,
            'start_time': start_time.isoformat() if start_time else None,
            'end_time': end_time.isoformat() if end_time else None,
            'status': training_status,
            'error_message': error_message
        }

        # Select the newly created model
        model_manager.load_model(model_id)

        # Clear the temporary data after successful training
        model_manager.clear_temp_data(data_id)
        
        print(f"Model training completed successfully!")
        print(f"   - Model ID: {model_id}")
        print(f"   - Model Name: {model_name}")
        print(f"   - Features used: {len(feature_cols)}")
        print(f"   - Model is trained: {trained_model.is_trained}")
        
        return {
            "message": "Model trained and saved successfully",
            "model_id": model_id,
            "model_name": model_name,
            "features_used": feature_cols,
            "target": target,
            "training_info": {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "samples_trained": X_values.shape[0],
                "feature_count": X_values.shape[1]
            }
        }
        
    except Exception as e:
        training_status = 'failed'
        error_message = str(e)
        model_manager.last_training_info = {
            'model_id': None,
            'start_time': start_time.isoformat() if start_time else None,
            'end_time': end_time.isoformat() if end_time else None,
            'status': training_status,
            'error_message': error_message
        }
        print(f"Training error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )

# Add new endpoints for model management
@app.get("/models")
async def list_models():
    """Get list of all available models."""
    try:
        models = model_manager.list_models()
        return {
            "models": models,
            "count": len(models),
            "current_model_id": model_manager.current_model_id
        }
    except Exception as e:
        print(f"Error listing models: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to list models: {str(e)}"
        )

@app.post("/models/select/{model_id}")
async def select_model(model_id: str):
    """Select a model by ID."""
    try:
        model_manager.load_model(model_id)
        current_model = model_manager.get_current_model()
        
        return {
            "message": f"Model {model_id} selected successfully",
            "model_id": model_id,
            "model_info": {
                "target_column": current_model['target_col'],
                "feature_count": len(current_model['feature_cols']),
                "feature_columns": current_model['feature_cols'],
                "is_trained": getattr(current_model['trained_model'], 'is_trained', False),
                "created_at": model_manager.models[model_id]['created_at'],
                "last_used": model_manager.models[model_id]['last_used']
            }
        }
    except Exception as e:
        print(f"Error selecting model {model_id}: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to select model: {str(e)}"
        )

@app.delete("/models/{model_id}")
async def delete_model_endpoint(model_id: str):
    """Delete a model by ID."""
    try:
        model_manager.delete_model(model_id)
        # Remove from status tracking
        if model_id in model_manager.model_status:
            del model_manager.model_status[model_id]
        return {"message": f"Model {model_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to delete model: {str(e)}"
        )

@app.get("/models/download/{model_id}")
async def download_model(model_id: str):
    """Download a model as a zip file containing model and metadata."""
    try:
        # Validate model exists
        if model_id not in model_manager.models:
            raise HTTPException(
                status_code=404, 
                detail=f"Model {model_id} not found"
            )
        
        # Validate model before download
        validation = model_manager.validate_model(model_id)
        if not validation['model_exists']:
            raise HTTPException(
                status_code=404, 
                detail=f"Model files not found for {model_id}"
            )
        
        # Create temporary zip file
        import tempfile
        import zipfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
            temp_zip_path = temp_file.name
        
        try:
            # Create zip file with model and metadata
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add model file
                model_file_path = model_manager.get_model_file_path(model_id)
                if os.path.exists(model_file_path):
                    zipf.write(model_file_path, f"{model_id}.pkl")
                
                # Add metadata file
                metadata_file_path = model_manager.get_model_metadata_file_path(model_id)
                if os.path.exists(metadata_file_path):
                    zipf.write(metadata_file_path, f"{model_id}_metadata.json")
                
                # Add validation report
                validation_info = {
                    'model_id': model_id,
                    'validation_result': validation,
                    'downloaded_at': datetime.now().isoformat(),
                    'status': model_manager.get_model_status(model_id)
                }
                
                zipf.writestr('validation_report.json', json.dumps(validation_info, indent=2))
            
            logger.info(f"Model {model_id} packaged for download")
            
            # Return file response
            return FileResponse(
                temp_zip_path,
                media_type='application/zip',
                filename=f"{model_id}_model_package.zip",
                background=BackgroundTasks().add_task(lambda: os.unlink(temp_zip_path))
            )
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_zip_path):
                os.unlink(temp_zip_path)
            raise e
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading model {model_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to download model: {str(e)}"
        )

@app.post("/models/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload and register a model from a zip file."""
    try:
        # Create temp file
        import tempfile
        import zipfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
        
        try:
            # Verify zip file
            if not zipfile.is_zipfile(temp_path):
                raise HTTPException(status_code=400, detail="Uploaded file is not a valid zip file")
            
            # Extract and process
            with zipfile.ZipFile(temp_path, 'r') as zipf:
                # Look for metadata file
                metadata_files = [f for f in zipf.namelist() if f.endswith('_metadata.json')]
                if not metadata_files:
                    raise HTTPException(status_code=400, detail="Invalid model package: metadata file missing")
                
                metadata_file = metadata_files[0]
                model_id = metadata_file.replace('_metadata.json', '')
                pkl_file = f"{model_id}.pkl"
                
                if pkl_file not in zipf.namelist():
                    raise HTTPException(status_code=400, detail="Invalid model package: model pickle file missing")
                
                # Extract files
                zipf.extract(metadata_file, model_manager.model_directory)
                zipf.extract(pkl_file, model_manager.model_directory)
                
                # Reload models
                model_manager.load_all_models()
                
                return {
                    "message": "Model uploaded successfully",
                    "model_id": model_id
                }
                
        finally:
            # Clean up uploaded zip
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload model: {str(e)}"
        )

# New visualization endpoints

@app.get("/training/status")
async def get_training_status():
    """Return the status of the current or last training session."""
    if model_manager.last_training_info is None:
        return {"message": "No training sessions recorded yet"}
    return model_manager.last_training_info

@app.get("/model/insights")
async def get_model_insights(model_id: str):
    """Return model insights including accuracy, feature importance, and other metrics."""
    if model_id not in model_manager.models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    try:
        state_data = model_manager.load_model(model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    cleaned_df = state_data['cleaned_df']
    trained_model = state_data['trained_model']
    feature_cols = state_data['feature_cols']
    target_col = state_data['target_col']

    y_true = cleaned_df[target_col].values
    X = cleaned_df[feature_cols].values
    y_pred = trained_model.predict(X).flatten()

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    feature_importance = {col: float(trained_model.weights[i][0]) for i, col in enumerate(feature_cols)}

    insights = {
        "model_id": model_id,
        "metrics": {"mse": mse, "r2_score": r2, "mae": mae},
        "feature_importance": feature_importance,
        "predictions": {"actual": y_true.tolist(), "predicted": y_pred.tolist()}
    }

    return insights

@app.get("/visualization/enhanced")
async def get_visualization_data(model_id: str):
    """Return enhanced visualization data in JSON format matching frontend VisualizationData interface."""
    if model_id not in model_manager.models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    try:
        state_data = model_manager.load_model(model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    cleaned_df = state_data['cleaned_df']
    trained_model = state_data['trained_model']
    feature_cols = state_data['feature_cols']
    target_col = state_data['target_col']

    # Get predictions and actual values
    y_true = cleaned_df[target_col].values
    X = cleaned_df[feature_cols].values
    y_pred = trained_model.predict(X).flatten()

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Residuals
    residuals = y_true - y_pred
    standardized_residuals = residuals / np.std(residuals) if np.std(residuals) > 0 else residuals

    # Feature importance (using absolute weights)
    feature_importance_scores = [float(abs(trained_model.weights[i][0])) for i in range(len(feature_cols))]
    coefficients = [float(trained_model.weights[i][0]) for i in range(len(feature_cols))]

    # Data distribution
    target_stats = cleaned_df[target_col].describe()
    target_quartiles = [target_stats['25%'], target_stats['50%'], target_stats['75%']]

    feature_distributions = {}
    for col in feature_cols:
        stats = cleaned_df[col].describe()
        feature_distributions[col] = {
            'mean': float(stats['mean']),
            'std': float(stats['std']),
            'min': float(stats['min']),
            'max': float(stats['max'])
        }

    # Correlation matrix
    corr_df = cleaned_df.corr()
    feature_names = list(corr_df.columns)
    correlation_values = corr_df.values.tolist()

    # Performance metrics (additional)
    explained_variance = 1 - np.var(residuals) / np.var(y_true) if np.var(y_true) > 0 else 0
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else 0

    # Data insights
    missing_values = cleaned_df.isnull().sum().to_dict()
    data_quality_score = 1 - (sum(missing_values.values()) / (len(cleaned_df) * len(cleaned_df.columns)))

    # Simple outlier detection (IQR method)
    outliers = {}
    for col in cleaned_df.select_dtypes(include=[np.number]).columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_indices = cleaned_df[(cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)].index.tolist()
        outliers[col] = outlier_indices

    feature_suggestions = ["Consider feature scaling if features have different ranges"]
    model_recommendations = ["Model performance looks good" if r2 > 0.8 else "Consider adding more features or using a different model"]

    return {
        "actual_vs_predicted": {
            "actual": y_true.tolist(),
            "predicted": y_pred.tolist(),
            "r2_score": float(r2),
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae)
        },
        "residuals": {
            "residuals": residuals.tolist(),
            "predicted_values": y_pred.tolist(),
            "standardized_residuals": standardized_residuals.tolist()
        },
        "feature_importance": {
            "feature_names": feature_cols,
            "importance_scores": feature_importance_scores,
            "coefficients": coefficients
        },
        "data_distribution": {
            "target_distribution": {
                "mean": float(target_stats['mean']),
                "median": float(target_stats['50%']),
                "std": float(target_stats['std']),
                "min": float(target_stats['min']),
                "max": float(target_stats['max']),
                "quartiles": [float(q) for q in target_quartiles]
            },
            "feature_distributions": feature_distributions
        },
        "correlation_matrix": {
            "feature_names": feature_names,
            "correlation_values": correlation_values
        },
        "performance_metrics": {
            "r2_score": float(r2),
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "explained_variance": float(explained_variance),
            "mape": float(mape)
        },
        "data_insights": {
            "data_quality_score": float(data_quality_score),
            "missing_values": {k: int(v) for k, v in missing_values.items()},
            "outliers": {k: [int(i) for i in v] for k, v in outliers.items()},
            "feature_suggestions": feature_suggestions,
            "model_recommendations": model_recommendations
        }
    }

@app.get("/training/recommendations")
async def get_training_recommendations():
    """Return recommendations for training."""
    return {
        "hyperparameters": {"learning_rate": "Start with 0.01, adjust based on convergence", "epochs": "1000 for small datasets, increase for larger ones"},
        "preprocessing": ["Ensure all data is numeric", "Handle missing values", "Normalize features if needed"],
        "model_types": ["Linear Regression for continuous targets", "Consider ensemble methods for complex data"]
    }

@app.get("/data/profile")
async def get_data_profile(model_id: str):
    """Return data profiling information for the model's training data matching DataProfile interface."""
    if model_id not in model_manager.models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    try:
        state_data = model_manager.load_model(model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    cleaned_df = state_data['cleaned_df']
    target_col = state_data['target_col']

    # Dataset info
    total_rows, total_columns = cleaned_df.shape
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    date_columns = []  # Assuming no date columns for simplicity

    # Data quality
    completeness_score = 1 - (cleaned_df.isnull().sum().sum() / (total_rows * total_columns))
    consistency_score = 1.0  # Placeholder
    uniqueness_score = cleaned_df.nunique().mean() / total_rows
    validity_score = 1.0  # Placeholder
    overall_score = (completeness_score + consistency_score + uniqueness_score + validity_score) / 4

    # Statistical summary
    numeric_summary = {}
    for col in numeric_columns:
        stats = cleaned_df[col].describe()
        numeric_summary[col] = {
            'count': int(stats['count']),
            'mean': float(stats['mean']),
            'std': float(stats['std']),
            'min': float(stats['min']),
            'q1': float(stats['25%']),
            'median': float(stats['50%']),
            'q3': float(stats['75%']),
            'max': float(stats['max']),
            'missing_count': int(cleaned_df[col].isnull().sum())
        }

    categorical_summary = {}
    for col in categorical_columns:
        value_counts = cleaned_df[col].value_counts()
        most_frequent = value_counts.index[0] if len(value_counts) > 0 else ""
        frequency = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
        categorical_summary[col] = {
            'unique_count': int(cleaned_df[col].nunique()),
            'most_frequent': most_frequent,
            'frequency': frequency,
            'missing_count': int(cleaned_df[col].isnull().sum())
        }

    # Correlations
    corr_df = cleaned_df.corr(numeric_only=True)
    high_correlations = []
    for i in range(len(corr_df.columns)):
        for j in range(i+1, len(corr_df.columns)):
            corr_val = corr_df.iloc[i, j]
            if abs(corr_val) > 0.7:  # High correlation threshold
                high_correlations.append({
                    'feature1': corr_df.columns[i],
                    'feature2': corr_df.columns[j],
                    'correlation': float(corr_val)
                })

    correlation_matrix = corr_df.values.tolist()
    feature_names = corr_df.columns.tolist()

    # Recommendations
    data_cleaning = []
    if completeness_score < 0.9:
        data_cleaning.append("Handle missing values")
    if len(high_correlations) > 0:
        data_cleaning.append("Consider removing highly correlated features")

    feature_engineering = ["Consider feature scaling" if len(numeric_columns) > 1 else ""]
    modeling = ["Linear regression is suitable for this data"]
    preprocessing = ["Ensure all data is numeric", "Handle outliers if present"]

    return {
        "dataset_info": {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "date_columns": date_columns,
            "target_column": target_col
        },
        "data_quality": {
            "completeness_score": float(completeness_score),
            "consistency_score": float(consistency_score),
            "uniqueness_score": float(uniqueness_score),
            "validity_score": float(validity_score),
            "overall_score": float(overall_score)
        },
        "statistical_summary": {
            "numeric_summary": numeric_summary,
            "categorical_summary": categorical_summary
        },
        "correlations": {
            "high_correlations": high_correlations,
            "correlation_matrix": correlation_matrix,
            "feature_names": feature_names
        },
        "recommendations": {
            "data_cleaning": data_cleaning,
            "feature_engineering": feature_engineering,
            "modeling": modeling,
            "preprocessing": preprocessing
        }
    }
