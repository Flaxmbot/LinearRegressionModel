import numpy as np
import pandas as pd
import sys
import logging

logger = logging.getLogger(__name__)

class Brain:
    def __init__(self, feature_size, action_size):
        self.weights = np.zeros((feature_size, action_size))
        self.bias = np.zeros(action_size)
        self.mean = None
        self.std = None
        self.prev_loss = float('inf')
        self.is_trained = False

    def _forward(self, features):
        return np.dot(features, self.weights) + self.bias

    def predict(self, features):
        features = np.array(features)
        if not self.is_trained or self.mean is None or self.std is None:
            raise ValueError("Model must be trained before making predictions")
        # Avoid division by zero
        std_safe = np.where(self.std == 0, 1, self.std)
        features = (features - self.mean) / std_safe
        return self._forward(features)
    
    def loss(self, target, features):
        target = np.array(target)
        # Use direct forward pass for loss calculation during training
        # This avoids the trained model check which would fail during training
        predictions = np.dot(features, self.weights) + self.bias
        return np.mean((predictions - target) ** 2)
    
    def gradient(self, target, features):
        N = features.shape[0]
        target = np.array(target).reshape(-1, self.bias.size)
        
        # Use direct forward pass for gradient calculation during training
        # This avoids the trained model check which would fail during training
        predictions = np.dot(features, self.weights) + self.bias
        error = predictions - target
        gradient = (2 / N ) * np.dot(features.T, error)
        bias_gradient = (2 / N) * np.sum(error, axis=0)

        return gradient, bias_gradient
    
    def update(self, gradient, bias_gradient, learning_rate):
        self.weights -= learning_rate * gradient
        self.bias -= learning_rate * bias_gradient

    def train(self, features, target, learning_rate, epochs):
        features = np.array(features)
        target = np.array(target)
        tolerance = 1e-5

        features_memory_mb = sys.getsizeof(features) / (1024 * 1024)
        target_memory_mb = sys.getsizeof(target) / (1024 * 1024)
        logger.info(f"Training started - features shape: {features.shape}, target shape: {target.shape}, features memory: {features_memory_mb:.2f} MB, target memory: {target_memory_mb:.2f} MB")
        
        # --- COMPREHENSIVE INPUT VALIDATION AND CONVERSION ---
        
        # 1. Check if features is a proper array
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # 2. Force conversion to numeric if possible
        if features.dtype == object:
            print("Warning: Features have object dtype, attempting conversion...")
            try:
                # Try to convert object array to numeric
                features_numeric = np.zeros_like(features, dtype=float)
                for i in range(features.shape[1]):
                    col = features[:, i]
                    try:
                        # Convert boolean values
                        if col.dtype == bool or (isinstance(col[0], bool)):
                            features_numeric[:, i] = col.astype(float)
                        else:
                            # Try numeric conversion
                            numeric_col = pd.to_numeric(col, errors='coerce')
                            if pd.isna(numeric_col).all():
                                # If all conversion failed, use 0
                                features_numeric[:, i] = 0
                            else:
                                features_numeric[:, i] = numeric_col.fillna(0).values
                    except:
                        # If conversion fails, use 0
                        features_numeric[:, i] = 0
                
                features = features_numeric
                print(f"Successfully converted object array to numeric. New dtype: {features.dtype}")
            except Exception as e:
                raise ValueError(f"Could not convert features to numeric: {e}")
        
        # 3. Additional validation
        if features.dtype == object:
            raise ValueError("Features array still has object dtype after conversion attempts. "
                             "Please ensure all data is numeric before training.")
        
        if features.ndim != 2:
            raise ValueError(f"Features must be a 2D array, got shape {features.shape}")
        
        if features.shape[0] == 0:
            raise ValueError("Features array is empty")
        
        if features.shape[1] == 0:
            raise ValueError("Features array has no columns")
        
        # 4. Check for and handle problematic values
        # Replace any remaining NaN or Inf values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 5. Ensure target is numeric
        if target.dtype == object:
            try:
                target = pd.to_numeric(target, errors='coerce').fillna(0).values
            except:
                target = np.zeros(len(target))
        
        # Final validation that numpy operations will work
        try:
            mean_test = np.mean(features, axis=0)
            std_test = np.std(features, axis=0)
            # print(f"SUCCESS: Numpy operations test passed. Features dtype: {features.dtype}, shape: {features.shape}")
        except Exception as e:
            raise ValueError(f"Numpy operations test failed: {e}")
        
        # --- CRITICAL FIX: RESHAPE TARGET ---
        # Ensure target is (N, 1) or (N, output_size) to match model output
        # This prevents the (N, N) broadcasting error in loss calculation
        target = target.reshape(-1, self.bias.size)

        # Calculate normalization parameters
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        # Avoid division by zero
        self.std = np.where(self.std == 0, 1, self.std)
        
        # Normalize features ONCE
        features_normalized = (features - self.mean) / self.std
        
        prev_loss = float('inf')  # Initialize prev_loss
        
        for epoch in range(epochs):
            # Gradient calculation
            grad, bias_grad = self.gradient(target, features_normalized)
            
            # Update weights
            self.update(grad, bias_grad, learning_rate)
            
            # Loss calculation
            # Now passing the reshaped target, so dimensions match correctly
            current_loss = self.loss(target, features_normalized)
            
            # Check for NaN/Explosion
            if np.isnan(current_loss) or np.isinf(current_loss):
                print(f"Training failed: Loss exploded at epoch {epoch}. Try reducing learning_rate.")
                break

            # Convergence Check
            if epoch > 0 and abs(prev_loss - current_loss) < tolerance:
                print(f"Converged early at epoch {epoch} (Loss: {current_loss:.6f})")
                break
            
            prev_loss = current_loss
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {current_loss:.6f}")
        
        self.is_trained = True