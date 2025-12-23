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
from typing import Optional, List, Dict, Any


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
        "http://127.0.0.1:5173"
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
            
            # Save model state
            state_data = {
                'cleaned_df': cleaned_df,
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
                'has_data': cleaned_df is not None,
                'feature_count': len(feature_cols),
                'target_column': target_col,
                'data_shape': cleaned_df.shape if cleaned_df is not None else None,
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
            print(f"⚠️  Warning: Could not load current model: {e}")
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
        
        print(f"✅ Prediction successful: {prediction.flatten().tolist()}")
        
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
        print(f"❌ Prediction error: {e}")
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
    
    print(f"📁 Processing file upload: {file.filename}")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        print(f"Original data shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        
        # Apply comprehensive data cleaning
        cleaned_data = comprehensive_data_cleaning(df)
        
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

        print(f"Training data prepared - Features: {len(feature_cols)}, Samples: {X_values.shape[0]}")
        print(f"Target column: {target}")
        print(f"Feature columns: {feature_cols}")

        # Create and train model
        trained_model = brain.Brain(feature_size=X_values.shape[1], action_size=1)

        print(f"Training model with {X_values.shape[1]} features...")
        trained_model.train(X_values, y, learning_rate=learning_rate, epochs=epochs)
        
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
            'data_id': data_id  # Store reference to original data
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
        if not file.filename.endswith('.zip'):
            raise HTTPException(
                status_code=400, 
                detail="Only zip files are supported for model upload"
            )
        
        # Save uploaded file temporarily
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
            temp_zip_path = temp_file.name
            contents = await file.read()
            temp_file.write(contents)
        
        try:
            # Extract and validate zip file
            with zipfile.ZipFile(temp_zip_path, 'r') as zipf:
                # List contents
                file_list = zipf.namelist()
                
                # Find model and metadata files
                model_file = None
                metadata_file = None
                
                for name in file_list:
                    if name.endswith('.pkl') and not name.endswith('_metadata.json'):
                        model_file = name
                    elif name.endswith('_metadata.json'):
                        metadata_file = name
                
                if not model_file:
                    raise HTTPException(
                        status_code=400, 
                        detail="No model file (.pkl) found in zip archive"
                    )
                
                # Extract files to temporary directory
                extract_dir = tempfile.mkdtemp()
                zipf.extractall(extract_dir)
                
                # Generate new model ID
                new_model_id = model_manager.generate_model_id()
                
                # Move files to models directory
                extracted_model_path = os.path.join(extract_dir, model_file)
                new_model_path = model_manager.get_model_file_path(new_model_id)
                
                # Copy model file
                import shutil
                shutil.copy2(extracted_model_path, new_model_path)
                
                # Handle metadata file if present
                if metadata_file:
                    extracted_metadata_path = os.path.join(extract_dir, metadata_file)
                    new_metadata_path = model_manager.get_model_metadata_file_path(new_model_id)
                    
                    # Update metadata with new model ID
                    with open(extracted_metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    metadata['model_id'] = new_model_id
                    metadata['uploaded_at'] = datetime.now().isoformat()
                    metadata['original_filename'] = file.filename
                    
                    with open(new_metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Add to models dictionary
                    model_manager.models[new_model_id] = metadata
                else:
                    # Create basic metadata
                    basic_metadata = {
                        'model_id': new_model_id,
                        'created_at': datetime.now().isoformat(),
                        'last_used': datetime.now().isoformat(),
                        'has_model': True,
                        'has_data': False,
                        'feature_count': 0,
                        'target_column': 'Unknown',
                        'data_shape': None,
                        'model_metadata': {
                            'uploaded_at': datetime.now().isoformat(),
                            'original_filename': file.filename
                        }
                    }
                    model_manager.models[new_model_id] = basic_metadata
                
                # Validate uploaded model
                validation = model_manager.validate_model(new_model_id)
                
                # Set status based on validation
                if validation['is_valid'] and validation['model_trained']:
                    model_manager.update_model_status(new_model_id, 'ready')
                elif validation['is_valid']:
                    model_manager.update_model_status(new_model_id, 'error')
                else:
                    model_manager.update_model_status(new_model_id, 'error')
                
                # Clean up
                shutil.rmtree(extract_dir)
                
                logger.info(f"Model uploaded successfully: {new_model_id}")
                
                return {
                    "message": "Model uploaded successfully",
                    "model_id": new_model_id,
                    "original_filename": file.filename,
                    "validation": validation,
                    "status": model_manager.get_model_status(new_model_id)
                }
                
        except zipfile.BadZipFile:
            raise HTTPException(
                status_code=400, 
                detail="Invalid zip file format"
            )
        finally:
            # Clean up temporary zip file
            if os.path.exists(temp_zip_path):
                os.unlink(temp_zip_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to upload model: {str(e)}"
        )

@app.get("/models/validate/{model_id}")
async def validate_model_endpoint(model_id: str):
    """Validate a model and return detailed validation results."""
    try:
        validation_result = model_manager.validate_model(model_id)
        return validation_result
    except Exception as e:
        logger.error(f"Error validating model {model_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to validate model: {str(e)}"
        )

@app.get("/status")
async def get_status():
    """Get current status of the model manager."""
    try:
        status = model_manager.get_status()
        
        # Add model status information
        status['model_statuses'] = {}
        for model_id in model_manager.models.keys():
            status['model_statuses'][model_id] = model_manager.get_model_status(model_id)
        
        # Add system information
        status['system_info'] = {
            'models_directory': model_manager.model_directory,
            'temp_data_count': len(model_manager.temp_data),
            'models_with_status': len(model_manager.model_status)
        }
        
        return status
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get status: {str(e)}"
        )

@app.get("/visualize")
def visualize_results():
    # Get current model data using the model manager
    current_model_data = model_manager.get_current_model()
    
    if not current_model_data:
        raise HTTPException(status_code=400, detail="No model selected. Please select a trained model first.")
    
    # Extract model data
    trained_model = current_model_data['trained_model']
    cleaned_df = current_model_data['cleaned_df']
    target_col = current_model_data['target_col']
    feature_cols = current_model_data['feature_cols']
    
    if trained_model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet. Please train the model first.")
    
    if cleaned_df is None:
        raise HTTPException(status_code=400, detail="No data available for visualization. Please upload and train a model first.")
    
    # Additional explicit DataFrame check
    if hasattr(cleaned_df, 'empty') and cleaned_df.empty:
        raise HTTPException(status_code=400, detail="Data is empty. Please upload and train a model first.")
    
    # Set Seaborn style for professional appearance
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'axes.spines.right': False,
        'axes.spines.top': False
    })
    sns.set_palette("husl")
    
    # Prepare data
    y_actual = cleaned_df[target_col].values
    X = cleaned_df.drop(columns=[target_col]).values
    y_pred = trained_model.predict(X)
    
    # Calculate model metrics
    mse = np.mean((y_actual - y_pred.flatten()) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_actual - y_pred.flatten()))
    r2 = 1 - (np.sum((y_actual - y_pred.flatten()) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2))
    
    # Create a comprehensive visualization dashboard
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Actual vs Predicted with regression line
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(x=y_actual, y=y_pred.flatten(), alpha=0.6, s=50, ax=ax1)
    sns.regplot(x=y_actual, y=y_pred.flatten(), scatter=False, color='red', ax=ax1)
    ax1.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 
             'k--', alpha=0.5, linewidth=1)
    ax1.set_xlabel(f'Actual {target_col}', fontsize=10, fontweight='bold')
    ax1.set_ylabel(f'Predicted {target_col}', fontsize=10, fontweight='bold')
    ax1.set_title('Actual vs Predicted Values', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add R² score to the plot
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
             fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Residuals plot
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = y_actual - y_pred.flatten()
    sns.scatterplot(x=y_pred.flatten(), y=residuals, alpha=0.6, s=50, ax=ax2)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel(f'Predicted {target_col}', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Residuals', fontsize=10, fontweight='bold')
    ax2.set_title('Residuals Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution of residuals
    ax3 = fig.add_subplot(gs[0, 2])
    sns.histplot(residuals, kde=True, alpha=0.7, ax=ax3)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Residuals', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax3.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature correlation heatmap
    ax4 = fig.add_subplot(gs[1, :2])
    
    # Create feature names if needed
    feature_names = feature_cols if len(feature_cols) > 0 else [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Ensure we have enough feature names
    if len(feature_names) < X.shape[1]:
        feature_names.extend([f'Feature_{i}' for i in range(len(feature_names), X.shape[1])])
    
    # Select a subset of features for correlation if too many
    max_features = min(10, X.shape[1])
    selected_features = feature_names[:max_features]
    selected_data = cleaned_df[selected_features + [target_col]]
    
    correlation_matrix = selected_data.corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax4)
    ax4.set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')
    
    # 5. Model performance metrics
    ax5 = fig.add_subplot(gs[1, 2])
    metrics = ['MSE', 'RMSE', 'MAE', 'R²']
    values = [mse, rmse, mae, r2]
    colors = sns.color_palette("viridis", len(metrics))
    
    bars = ax5.bar(metrics, values, color=colors, alpha=0.8)
    ax5.set_title('Model Performance Metrics', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Value', fontsize=10, fontweight='bold')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Feature importance (using absolute weights)
    ax6 = fig.add_subplot(gs[2, :])
    
    if len(trained_model.weights) > 0:
        # Get feature importance (absolute weights)
        feature_importance = np.abs(trained_model.weights.flatten())
        
        # Ensure we have feature names
        if len(feature_names) != len(feature_importance):
            # Adjust feature names to match importance array
            feature_names = feature_names[:len(feature_importance)]
            if len(feature_names) < len(feature_importance):
                feature_names.extend([f'Feature_{i}' for i in range(len(feature_names), len(feature_importance))])
        
        # Sort by importance
        sorted_indices = np.argsort(feature_importance)[::-1]
        sorted_importance = feature_importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # Limit to top 15 features for readability
        top_n = min(15, len(sorted_importance))
        top_importance = sorted_importance[:top_n]
        top_names = sorted_names[:top_n]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_names))
        bars = ax6.barh(y_pos, top_importance, color=sns.color_palette("plasma", len(top_importance)), alpha=0.8)
        
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(top_names, fontsize=8)
        ax6.set_xlabel('Feature Importance (|Weight|)', fontsize=10, fontweight='bold')
        ax6.set_title('Top Feature Importance', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_importance)):
            width = bar.get_width()
            ax6.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{value:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'No weights available for feature importance', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Feature Importance', fontsize=12, fontweight='bold')
    
    # Add overall title
    fig.suptitle('ML Model Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    # Add model info text
    info_text = f"""Model: Simple Neural Network
    Features: {len(feature_cols)}
    Target: {target_col}
    Data Points: {len(cleaned_df)}"""
    
    fig.text(0.02, 0.02, info_text, fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return StreamingResponse(buf, media_type="image/png")

# ==========================================
# MISSING API ENDPOINTS IMPLEMENTATION
# ==========================================

@app.get("/training/status")
async def get_all_training_statuses():
    """Get status of all training processes"""
    try:
        # Get all model statuses
        all_statuses = []
        for model_id in model_manager.models.keys():
            status_info = model_manager.get_model_status(model_id)
            if status_info:
                # Get additional model info
                model_info = model_manager.models.get(model_id, {})
                status_with_info = {
                    'model_id': model_id,
                    'status': status_info.get('status', 'unknown'),
                    'updated_at': status_info.get('updated_at'),
                    'model_name': model_info.get('model_name', 'Unknown'),
                    'target_column': model_info.get('target_column', 'Unknown'),
                    'created_at': model_info.get('created_at'),
                    'training_samples': model_info.get('training_samples', 0),
                    'feature_count': model_info.get('feature_count', 0)
                }
                all_statuses.append(status_with_info)
        
        return {
            "training_statuses": all_statuses,
            "total_models": len(all_statuses),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting training statuses: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get training statuses: {str(e)}"
        )

@app.get("/training/status/{model_id}")
async def get_training_status(model_id: str):
    """Get training status for specific model"""
    try:
        # Check if model exists
        if model_id not in model_manager.models:
            raise HTTPException(
                status_code=404, 
                detail=f"Model {model_id} not found"
            )
        
        # Get model status
        status_info = model_manager.get_model_status(model_id)
        model_info = model_manager.models[model_id]
        
        if not status_info:
            # Model exists but no status tracking - assume ready
            status_info = {
                'status': 'ready',
                'updated_at': datetime.now().isoformat()
            }
        
        # Get current model data if it's the active one
        current_data = None
        if model_manager.current_model_id == model_id:
            try:
                current_data = model_manager.get_current_model()
            except:
                pass
        
        # Calculate training progress (mock data for now)
        progress_info = {
            'progress': 100 if status_info.get('status') == 'ready' else 0,
            'message': f"Model is {status_info.get('status', 'unknown')}",
            'current_epoch': None,
            'total_epochs': None,
            'loss': None,
            'validation_loss': None
        }
        
        return {
            "model_id": model_id,
            "status": status_info.get('status', 'unknown'),
            "updated_at": status_info.get('updated_at'),
            "model_info": {
                "model_name": model_info.get('model_name', 'Unknown'),
                "target_column": model_info.get('target_column', 'Unknown'),
                "feature_count": model_info.get('feature_count', 0),
                "training_samples": model_info.get('training_samples', 0),
                "created_at": model_info.get('created_at'),
                "has_model": model_info.get('has_model', False),
                "has_data": model_info.get('has_data', False)
            },
            "training_progress": progress_info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training status for {model_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get training status: {str(e)}"
        )

@app.get("/visualization/enhanced")
async def get_enhanced_visualization():
    """Get comprehensive visualization data as JSON"""
    try:
        # Get current model data
        current_model_data = model_manager.get_current_model()
        
        if not current_model_data:
            raise HTTPException(
                status_code=400, 
                detail="No model selected. Please select a trained model first."
            )
        
        trained_model = current_model_data['trained_model']
        cleaned_df = current_model_data['cleaned_df']
        target_col = current_model_data['target_col']
        feature_cols = current_model_data['feature_cols']
        
        if trained_model is None:
            raise HTTPException(
                status_code=400, 
                detail="Model not trained yet. Please train the model first."
            )
        
        if cleaned_df is None:
            raise HTTPException(
                status_code=400, 
                detail="No data available for visualization. Please upload and train a model first."
            )
        
        # Additional explicit DataFrame check
        if hasattr(cleaned_df, 'empty') and cleaned_df.empty:
            raise HTTPException(
                status_code=400, 
                detail="Data is empty. Please upload and train a model first."
            )
        
        # Prepare data
        y_actual = cleaned_df[target_col].values
        X = cleaned_df.drop(columns=[target_col]).values
        y_pred = trained_model.predict(X)
        
        # Calculate model metrics
        mse = float(np.mean((y_actual - y_pred.flatten()) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_actual - y_pred.flatten())))
        r2 = float(1 - (np.sum((y_actual - y_pred.flatten()) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2)))
        
        # Actual vs Predicted data
        actual_vs_predicted = {
            "actual": y_actual.tolist(),
            "predicted": y_pred.flatten().tolist(),
            "r2_score": r2,
            "mse": mse,
            "rmse": rmse,
            "mae": mae
        }
        
        # Residuals data
        residuals = y_actual - y_pred.flatten()
        residuals_data = {
            "residuals": residuals.tolist(),
            "predicted_values": y_pred.flatten().tolist(),
            "standardized_residuals": ((residuals - np.mean(residuals)) / np.std(residuals)).tolist()
        }
        
        # Feature importance (using absolute weights)
        feature_importance_data = {
            "feature_names": feature_cols,
            "importance_scores": np.abs(trained_model.weights.flatten()).tolist() if len(trained_model.weights) > 0 else [],
            "coefficients": trained_model.weights.flatten().tolist() if len(trained_model.weights) > 0 else []
        }
        
        # Correlation matrix (for selected features)
        max_features = min(10, len(feature_cols))
        selected_features = feature_cols[:max_features] + [target_col]
        selected_data = cleaned_df[selected_features]
        correlation_matrix = selected_data.corr().fillna(0)
        
        # Response structure matching frontend expectations
        return {
            "actual_vs_predicted": actual_vs_predicted,
            "residuals": residuals_data,
            "feature_importance": feature_importance_data,
            "correlation_matrix": {
                "features": selected_features,
                "matrix": correlation_matrix.values.tolist()
            },
            "model_metrics": {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2_score": r2
            },
            "dataset_info": {
                "total_samples": len(cleaned_df),
                "feature_count": len(feature_cols),
                "target_column": target_col
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting enhanced visualization: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get enhanced visualization: {str(e)}"
        )

@app.get("/visualization/chart/{chartType}")
async def get_specific_chart(chartType: str):
    """Get data for specific chart type"""
    try:
        # Get current model data
        current_model_data = model_manager.get_current_model()
        
        if not current_model_data:
            raise HTTPException(
                status_code=400, 
                detail="No model selected. Please select a trained model first."
            )
        
        trained_model = current_model_data['trained_model']
        cleaned_df = current_model_data['cleaned_df']
        target_col = current_model_data['target_col']
        feature_cols = current_model_data['feature_cols']
        
        if trained_model is None:
            raise HTTPException(
                status_code=400, 
                detail="Model not trained yet. Please train the model first."
            )
        
        if cleaned_df is None:
            raise HTTPException(
                status_code=400, 
                detail="No data available. Please upload and train a model first."
            )
        
        # Additional explicit DataFrame check
        if hasattr(cleaned_df, 'empty') and cleaned_df.empty:
            raise HTTPException(
                status_code=400, 
                detail="Data is empty. Please upload and train a model first."
            )
        
        # Prepare data
        y_actual = cleaned_df[target_col].values
        X = cleaned_df.drop(columns=[target_col]).values
        y_pred = trained_model.predict(X)
        
        # Handle different chart types
        if chartType.lower() == "actualvspredicted" or chartType.lower() == "actualvsPredicted":
            # Calculate metrics
            mse = float(np.mean((y_actual - y_pred.flatten()) ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(y_actual - y_pred.flatten())))
            r2 = float(1 - (np.sum((y_actual - y_pred.flatten()) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2)))
            
            return {
                "chart_type": "actual_vs_predicted",
                "data": {
                    "actual": y_actual.tolist(),
                    "predicted": y_pred.flatten().tolist()
                },
                "metrics": {
                    "r2_score": r2,
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae
                },
                "axis_labels": {
                    "x": f"Actual {target_col}",
                    "y": f"Predicted {target_col}"
                }
            }
        
        elif chartType.lower() == "residuals":
            residuals = y_actual - y_pred.flatten()
            return {
                "chart_type": "residuals",
                "data": {
                    "predicted": y_pred.flatten().tolist(),
                    "residuals": residuals.tolist()
                },
                "axis_labels": {
                    "x": f"Predicted {target_col}",
                    "y": "Residuals"
                }
            }
        
        elif chartType.lower() == "featureimportance" or chartType.lower() == "featureImportance":
            if len(trained_model.weights) == 0:
                return {
                    "chart_type": "feature_importance",
                    "data": {
                        "features": [],
                        "importance": []
                    },
                    "message": "No feature weights available"
                }
            
            # Get feature importance (absolute weights)
            feature_importance = np.abs(trained_model.weights.flatten())
            
            # Ensure we have feature names
            feature_names = feature_cols[:len(feature_importance)]
            if len(feature_names) < len(feature_importance):
                feature_names.extend([f'Feature_{i}' for i in range(len(feature_names), len(feature_importance))])
            
            return {
                "chart_type": "feature_importance",
                "data": {
                    "features": feature_names,
                    "importance": feature_importance.tolist()
                },
                "axis_labels": {
                    "x": "Features",
                    "y": "Importance (|Weight|)"
                }
            }
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown chart type: {chartType}. Supported types: actualVsPredicted, residuals, featureImportance"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chart data for {chartType}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get chart data: {str(e)}"
        )

@app.get("/data/profile")
async def get_data_profile():
    """Get comprehensive data analysis and profiling"""
    try:
        # Get current model data
        current_model_data = model_manager.get_current_model()
        
        if not current_model_data:
            raise HTTPException(
                status_code=400, 
                detail="No model selected. Please select a trained model first."
            )
        
        cleaned_df = current_model_data['cleaned_df']
        
        if cleaned_df is None:
            raise HTTPException(
                status_code=400, 
                detail="No data available for profiling."
            )
        
        # Dataset info
        numeric_columns = cleaned_df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = cleaned_df.select_dtypes(include=['object']).columns.tolist()
        
        # Data quality metrics
        total_rows, total_columns = cleaned_df.shape
        
        # Completeness (missing values)
        missing_counts = cleaned_df.isnull().sum()
        completeness_scores = {}
        for col in cleaned_df.columns:
            completeness_scores[col] = float(1 - (missing_counts[col] / total_rows))
        
        overall_completeness = np.mean(list(completeness_scores.values()))
        
        # Consistency (basic variance check for numeric columns)
        consistency_scores = {}
        for col in numeric_columns:
            if cleaned_df[col].nunique() > 1:
                consistency_scores[col] = 1.0  # Good consistency if has variance
            else:
                consistency_scores[col] = 0.0  # Poor consistency if no variance
        
        overall_consistency = np.mean(list(consistency_scores.values())) if consistency_scores else 1.0
        
        # Uniqueness (duplicate rows)
        duplicate_rows = cleaned_df.duplicated().sum()
        uniqueness_score = float(1 - (duplicate_rows / total_rows))
        
        # Validity (basic range check for numeric columns)
        validity_scores = {}
        for col in numeric_columns:
            # Check for extreme outliers (beyond 3 standard deviations)
            if cleaned_df[col].std() > 0:
                z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
                outlier_ratio = (z_scores > 3).sum() / len(cleaned_df)
                validity_scores[col] = float(1 - outlier_ratio)
            else:
                validity_scores[col] = 1.0
        
        overall_validity = np.mean(list(validity_scores.values())) if validity_scores else 1.0
        
        # Overall score
        overall_score = float((overall_completeness + overall_consistency + uniqueness_score + overall_validity) / 4)
        
        # Statistical summary
        numeric_summary = {}
        for col in numeric_columns:
            series = cleaned_df[col]
            numeric_summary[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75))
            }
        
        # Categorical summary
        categorical_summary = {}
        for col in categorical_columns:
            value_counts = cleaned_df[col].value_counts().head(10)  # Top 10 values
            categorical_summary[col] = {
                "unique_count": int(cleaned_df[col].nunique()),
                "top_values": value_counts.to_dict(),
                "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None
            }
        
        return {
            "dataset_info": {
                "total_rows": total_rows,
                "total_columns": total_columns,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "memory_usage_mb": float(cleaned_df.memory_usage(deep=True).sum() / 1024 / 1024)
            },
            "data_quality": {
                "completeness_score": float(overall_completeness),
                "consistency_score": float(overall_consistency),
                "uniqueness_score": float(uniqueness_score),
                "validity_score": float(overall_validity),
                "overall_score": float(overall_score),
                "completeness_by_column": completeness_scores,
                "consistency_by_column": consistency_scores,
                "validity_by_column": validity_scores
            },
            "statistics": {
                "numeric_summary": numeric_summary,
                "categorical_summary": categorical_summary
            },
            "generated_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data profile: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get data profile: {str(e)}"
        )

@app.get("/model/insights")
async def get_model_insights():
    """Get model performance insights and recommendations"""
    try:
        # Get current model data
        current_model_data = model_manager.get_current_model()
        
        if not current_model_data:
            raise HTTPException(
                status_code=400, 
                detail="No model selected. Please select a trained model first."
            )
        
        trained_model = current_model_data['trained_model']
        cleaned_df = current_model_data['cleaned_df']
        target_col = current_model_data['target_col']
        feature_cols = current_model_data['feature_cols']
        
        if trained_model is None:
            raise HTTPException(
                status_code=400, 
                detail="Model not trained yet. Please train the model first."
            )
        
        if cleaned_df is None:
            raise HTTPException(
                status_code=400, 
                detail="No data available for insights. Please upload and train a model first."
            )
        
        # Additional explicit DataFrame check
        if hasattr(cleaned_df, 'empty') and cleaned_df.empty:
            raise HTTPException(
                status_code=400, 
                detail="Data is empty. Please upload and train a model first."
            )
        
        # Prepare data
        y_actual = cleaned_df[target_col].values
        X = cleaned_df.drop(columns=[target_col]).values
        y_pred = trained_model.predict(X)
        
        # Calculate metrics
        mse = float(np.mean((y_actual - y_pred.flatten()) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_actual - y_pred.flatten())))
        r2 = float(1 - (np.sum((y_actual - y_pred.flatten()) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2)))
        
        # Generate insights
        insights = {
            "model_performance": {
                "overall_score": float(r2),
                "r2_score": r2,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "performance_grade": "Excellent" if r2 > 0.9 else "Good" if r2 > 0.7 else "Fair" if r2 > 0.5 else "Poor"
            },
            "model_characteristics": {
                "model_type": "Simple Neural Network",
                "feature_count": len(feature_cols),
                "sample_count": len(cleaned_df),
                "target_variable": target_col,
                "has_weights": len(trained_model.weights) > 0
            },
            "feature_insights": {},
            "recommendations": []
        }
        
        # Feature importance analysis
        if len(trained_model.weights) > 0:
            feature_importance = np.abs(trained_model.weights.flatten())
            sorted_indices = np.argsort(feature_importance)[::-1]
            
            # Top 5 most important features
            top_features = []
            for i in range(min(5, len(sorted_indices))):
                idx = sorted_indices[i]
                feature_name = feature_cols[idx] if idx < len(feature_cols) else f"Feature_{idx}"
                importance = float(feature_importance[idx])
                top_features.append({
                    "feature": feature_name,
                    "importance": importance,
                    "rank": i + 1
                })
            
            insights["feature_insights"] = {
                "top_features": top_features,
                "importance_distribution": {
                    "mean_importance": float(np.mean(feature_importance)),
                    "std_importance": float(np.std(feature_importance)),
                    "max_importance": float(np.max(feature_importance)),
                    "min_importance": float(np.min(feature_importance))
                }
            }
        
        # Performance-based recommendations
        recommendations = []
        
        if r2 < 0.5:
            recommendations.append({
                "category": "Model Performance",
                "priority": "High",
                "suggestion": "Consider feature engineering or trying different algorithms. Current R² is low."
            })
        
        if r2 > 0.9:
            recommendations.append({
                "category": "Model Validation",
                "priority": "Medium",
                "suggestion": "High R² may indicate overfitting. Consider cross-validation."
            })
        
        if len(feature_cols) > 20:
            recommendations.append({
                "category": "Feature Selection",
                "priority": "Medium",
                "suggestion": "Large number of features may lead to overfitting. Consider feature selection."
            })
        
        if len(cleaned_df) < 100:
            recommendations.append({
                "category": "Data Volume",
                "priority": "High",
                "suggestion": "Small dataset size may limit model performance. Consider collecting more data."
            })
        
        # Feature-specific recommendations
        if len(trained_model.weights) > 0:
            low_importance_features = []
            feature_importance = np.abs(trained_model.weights.flatten())
            threshold = np.mean(feature_importance) * 0.1  # Features with < 10% of average importance
            
            for i, importance in enumerate(feature_importance):
                if importance < threshold:
                    feature_name = feature_cols[i] if i < len(feature_cols) else f"Feature_{i}"
                    low_importance_features.append(feature_name)
            
            if low_importance_features:
                recommendations.append({
                    "category": "Feature Importance",
                    "priority": "Low",
                    "suggestion": f"Features with low importance: {', '.join(low_importance_features[:5])}. Consider removing them."
                })
        
        insights["recommendations"] = recommendations
        
        return insights
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model insights: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get model insights: {str(e)}"
        )

@app.get("/training/recommendations")
async def get_training_recommendations():
    """Get training parameter recommendations"""
    try:
        # Get current model data
        current_model_data = model_manager.get_current_model()
        
        if not current_model_data:
            raise HTTPException(
                status_code=400, 
                detail="No model selected. Please select a trained model first."
            )
        
        cleaned_df = current_model_data['cleaned_df']
        feature_cols = current_model_data['feature_cols']
        
        if cleaned_df is None:
            raise HTTPException(
                status_code=400, 
                detail="No data available for training recommendations. Please upload and train a model first."
            )
        
        # Additional explicit DataFrame check
        if hasattr(cleaned_df, 'empty') and cleaned_df.empty:
            raise HTTPException(
                status_code=400, 
                detail="Data is empty. Please upload and train a model first."
            )
        
        # Analyze data characteristics
        n_samples = len(cleaned_df)
        n_features = len(feature_cols)
        
        # Basic dataset statistics
        numeric_columns = cleaned_df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = cleaned_df.select_dtypes(include=['object']).columns.tolist()
        
        # Generate recommendations
        recommendations = {
            "dataset_analysis": {
                "sample_size": n_samples,
                "feature_count": n_features,
                "numeric_features": len(numeric_columns),
                "categorical_features": len(categorical_columns),
                "data_complexity": "High" if n_features > 20 else "Medium" if n_features > 5 else "Low"
            },
            "training_suggestions": {
                "learning_rate": {
                    "recommended": 0.01 if n_samples > 1000 else 0.001,
                    "range": [0.0001, 0.1],
                    "rationale": "Smaller learning rate for smaller datasets to ensure stability"
                },
                "epochs": {
                    "recommended": min(1000, max(100, n_samples // 10)),
                    "range": [50, 5000],
                    "rationale": "Based on dataset size - more samples need fewer epochs"
                },
                "batch_size": {
                    "recommended": min(32, max(8, n_samples // 10)),
                    "range": [8, 128],
                    "rationale": "Adaptive batch size based on dataset size"
                }
            },
            "preprocessing_recommendations": [],
            "optimization_tips": []
        }
        
        # Preprocessing recommendations
        if len(categorical_columns) > 0:
            recommendations["preprocessing_recommendations"].append({
                "issue": "Categorical Features",
                "suggestion": f"Consider encoding categorical columns: {categorical_columns[:5]}",
                "impact": "High",
                "methods": ["One-hot encoding", "Label encoding", "Target encoding"]
            })
        
        # Check for potential overfitting indicators
        if n_features > n_samples * 0.2:
            recommendations["optimization_tips"].append({
                "issue": "High Feature-to-Sample Ratio",
                "suggestion": "Consider feature selection or regularization",
                "priority": "High"
            })
        
        # Check for underfitting indicators
        if n_features < 3 and n_samples > 1000:
            recommendations["optimization_tips"].append({
                "issue": "Low Feature Count",
                "suggestion": "Consider adding more features or using polynomial features",
                "priority": "Medium"
            })
        
        # Missing values check
        missing_ratios = cleaned_df.isnull().sum() / len(cleaned_df)
        high_missing = missing_ratios[missing_ratios > 0.1]
        if len(high_missing) > 0:
            recommendations["preprocessing_recommendations"].append({
                "issue": "Missing Values",
                "suggestion": f"Columns with high missing values: {list(high_missing.index[:5])}",
                "impact": "Medium",
                "methods": ["Imputation", "Drop rows/columns", "Use models that handle missing values"]
            })
        
        # Standardization recommendations
        if len(numeric_columns) > 0:
            recommendations["preprocessing_recommendations"].append({
                "issue": "Feature Scaling",
                "suggestion": "Consider standardizing numeric features",
                "impact": "Medium",
                "methods": ["StandardScaler", "MinMaxScaler", "RobustScaler"]
            })
        
        # Cross-validation recommendation
        if n_samples < 1000:
            recommendations["optimization_tips"].append({
                "issue": "Cross-Validation",
                "suggestion": "Use cross-validation to get reliable performance estimates",
                "priority": "High"
            })
        
        return recommendations
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training recommendations: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get training recommendations: {str(e)}"
        )

