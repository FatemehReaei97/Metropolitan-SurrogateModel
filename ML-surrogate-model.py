import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import time
import pickle
from typing import Dict, Tuple
from pathlib import Path
import sys

class RFModelTrainer:
    """Random Forest model trainer for flood prediction.
    
    This class handles:
    1. Data loading and preprocessing
    2. Model training with GridSearchCV
    3. Feature importance analysis
    4. Model validation and testing
    5. Performance visualization
    6. Runtime tracking
    """
    
    def __init__(self, cluster_id: int, base_path: str):
        """Initialize trainer for a specific cluster.
        
        Args:
            cluster_id: ID of the cluster being processed
            base_path: Base path for data files
        """
        self.cluster_id = cluster_id
        self.base_path = Path(base_path)
        self.output_path = self.base_path / 'results' / f'cluster_{cluster_id}'
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Model hyperparameters for grid search
        self.rf_params = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [None, 5, 10],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_features': ['auto', 'sqrt'],
            'regressor__bootstrap': [True, False]
        }

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load training and test data."""
        try:
            # Training data (synthetic rainfall events)
            features_df = pd.read_excel(
                self.base_path / f'training_features_{self.cluster_id}.xlsx', 
                header=0
            )
            target_df = pd.read_excel(
                self.base_path / f'training_targets_{self.cluster_id}.xlsx'
            )
            
            # Test data (real rainfall events)
            features_test = pd.read_excel(
                self.base_path / f'test_features_{self.cluster_id}.xlsx', 
                header=0
            )
            target_test = pd.read_excel(
                self.base_path / f'test_targets_{self.cluster_id}.xlsx'
            )
            
            return features_df, target_df, features_test, target_test
            
        except Exception as e:
            print(f"Error loading data for cluster {self.cluster_id}: {str(e)}")
            raise

    def train_model(self) -> Tuple[Pipeline, Dict[str, float]]:
        """Train model and track runtime for different phases."""
        runtimes = {}
        total_start = time.time()
        
        # Track data loading time
        data_start = time.time()
        features_df, target_df, features_test, target_test = self.load_data()
        X = features_df.iloc[:, 1:]
        y = target_df.iloc[:, 1:].T
        
        # Split data randomly as we're not dealing with time series prediction
        # Each sample is an independent event (rainfall-flood relationship)
        # so we can shuffle the data without losing temporal dependencies
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        runtimes['data_loading'] = time.time() - data_start
        
        # Track model training time
        train_start = time.time()
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('regressor', RandomForestRegressor())
        ])
        
        grid = GridSearchCV(
            pipeline, 
            self.rf_params, 
            scoring='neg_mean_squared_error',
            cv=5
        )
        grid.fit(X_train, y_train)
        runtimes['model_training'] = time.time() - train_start
        
        # Evaluate model
        val_start = time.time()
        self.evaluate_model(grid.best_estimator_, X_val, y_val, "validation")
        runtimes['validation'] = time.time() - val_start
        
        test_start = time.time()
        self.evaluate_model(grid.best_estimator_, features_test.iloc[:, 1:], 
                          target_test.iloc[:, 1:].T, "test")
        runtimes['test'] = time.time() - test_start
        
        # Analyze feature importance
        self.analyze_feature_importance(grid.best_estimator_, X)
        
        # Total runtime
        runtimes['total'] = time.time() - total_start
        
        # Save runtime information
        self._save_runtime_info(runtimes)
        
        return grid.best_estimator_, runtimes

    def analyze_feature_importance(self, model: Pipeline, X: pd.DataFrame):
        """Analyze and save feature importance."""
        importances = model.named_steps['regressor'].feature_importances_
        feature_importance = list(zip(X.columns, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        with open(self.output_path / f'feature_importance_{self.cluster_id}.txt', 'w') as f:
            f.write(f"Feature Importance Analysis for Cluster {self.cluster_id}\n")
            f.write("=" * 50 + "\n\n")
            for feature, importance in feature_importance:
                f.write(f"{feature}: {importance:.4f}\n")

    def evaluate_model(self, model: Pipeline, X: pd.DataFrame, 
                      y: pd.DataFrame, set_name: str):
        """Evaluate model performance."""
        predictions = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        
        # Save metrics
        with open(self.output_path / f'metrics_{set_name}_{self.cluster_id}.txt', 'w') as f:
            f.write(f"Performance Metrics for Cluster {self.cluster_id} - {set_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"R²: {r2:.4f}\n")
        
        # Plot results
        self._plot_results(y, predictions, set_name)
        
        return rmse, r2

    def _plot_results(self, y_true: pd.DataFrame, y_pred: np.ndarray, set_name: str):
        """Create and save prediction plots."""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, c='blue', label='Predicted vs. Observed')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('Observed SWMM Results')
        plt.ylabel('Predicted Random Forest')
        plt.title(f'Cluster {self.cluster_id} - {set_name} Set Predictions')
        plt.legend()
        plt.savefig(self.output_path / f'predictions_{set_name}_{self.cluster_id}.png')
        plt.close()

    def _save_runtime_info(self, runtimes: Dict[str, float]):
        """Save detailed runtime information."""
        runtime_file = self.output_path / f'runtime_analysis_{self.cluster_id}.txt'
        with open(runtime_file, 'w') as f:
            f.write(f"Runtime Analysis for Cluster {self.cluster_id}\n")
            f.write("=" * 50 + "\n\n")
            f.write("Detailed Breakdown:\n")
            f.write(f"Data Loading Time: {runtimes['data_loading']:.2f} seconds\n")
            f.write(f"Model Training Time: {runtimes['model_training']:.2f} seconds\n")
            f.write(f"Validation Time: {runtimes['validation']:.2f} seconds\n")
            f.write(f"Test Time: {runtimes['test']:.2f} seconds\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Runtime: {runtimes['total']:.2f} seconds\n")

def main():
    """Main execution function for HPC environment."""
    if len(sys.argv) != 2:
        print("Usage: python random_forest_trainer.py <cluster_id>")
        sys.exit(1)
        
    cluster_id = int(sys.argv[1])
    trainer = RFModelTrainer(cluster_id, '/path/to/data')
    
    try:
        # Train model and get runtimes
        model, runtimes = trainer.train_model()
        
        # Save model
        with open(trainer.output_path / f'model_{cluster_id}.pkl', 'wb') as f:
            pickle.dump(model, f)
            
    except Exception as e:
        print(f"Error processing cluster {cluster_id}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()