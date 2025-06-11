"""
Model Trainer Module for NBE Prediction Project
Handles training of baseline and enhanced models with multiple algorithms
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
from datetime import datetime
import pickle
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings

# Handle XGBoost import gracefully
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Handles training of multiple ML models for both baseline and enhanced feature sets
    """

    def __init__(self, models_path: Path, log_path: Path):
        self.models_path = models_path
        self.log_path = log_path
        self.logger = self._setup_logger()

        # Random state for reproducibility
        self.random_state = 42

        # Initialize model configurations
        self._setup_model_configs()

        # Create models directory
        self.artifacts_dir = self.models_path / 'artifacts'
        self.metadata_dir = self.models_path / 'metadata'
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for model training operations"""
        logger = logging.getLogger('ModelTrainer')
        logger.setLevel(logging.INFO)

        # Create log directory
        log_dir = self.log_path / 'step4'
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'model_trainer_{timestamp}.log'

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(file_handler)

        return logger

    def _setup_model_configs(self):
        """Setup model configurations with default hyperparameters"""
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    solver='liblinear'  # Good for small datasets
                ),
                'name': 'Logistic Regression',
                'type': 'linear'
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2
                ),
                'name': 'Random Forest',
                'type': 'ensemble'
            }
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.model_configs['xgboost'] = {
                'model': XGBClassifier(
                    random_state=self.random_state,
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    eval_metric='logloss'
                ),
                'name': 'XGBoost',
                'type': 'boosting'
            }

    def load_training_data(self, data_path: Path) -> Dict[str, pd.DataFrame]:
        """
        Load both baseline and enhanced datasets

        Args:
            data_path: Path to processed data directory

        Returns:
            Dict containing all training datasets
        """
        self.logger.info("Loading training datasets")

        processed_dir = data_path / 'processed'

        # Find most recent files
        baseline_train_files = list(processed_dir.glob('step2_baseline_train_*.csv'))
        baseline_test_files = list(processed_dir.glob('step2_baseline_test_*.csv'))
        enhanced_train_files = list(processed_dir.glob('step2_enhanced_train_*.csv'))
        enhanced_test_files = list(processed_dir.glob('step2_enhanced_test_*.csv'))

        if not all([baseline_train_files, baseline_test_files, enhanced_train_files, enhanced_test_files]):
            raise FileNotFoundError("Missing required dataset files from Step 2")

        # Load most recent files
        datasets = {
            'baseline_train': pd.read_csv(max(baseline_train_files, key=lambda x: x.stat().st_mtime)),
            'baseline_test': pd.read_csv(max(baseline_test_files, key=lambda x: x.stat().st_mtime)),
            'enhanced_train': pd.read_csv(max(enhanced_train_files, key=lambda x: x.stat().st_mtime)),
            'enhanced_test': pd.read_csv(max(enhanced_test_files, key=lambda x: x.stat().st_mtime))
        }

        # Log dataset info
        for name, df in datasets.items():
            self.logger.info(f"{name}: {df.shape[0]} records, {df.shape[1] - 1} features")

        return datasets

    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable

        Args:
            df: Dataset with features and target

        Returns:
            Tuple of features and target
        """
        if 'nbe' not in df.columns:
            raise ValueError("Target variable 'nbe' not found in dataset")

        X = df.drop('nbe', axis=1)
        y = df['nbe']

        return X, y

    def train_single_model(self, model_config: Dict, X_train: pd.DataFrame,
                           y_train: pd.Series, model_type: str) -> Dict[str, Any]:
        """
        Train a single model and return training results

        Args:
            model_config: Model configuration dictionary
            X_train: Training features
            y_train: Training target
            model_type: 'baseline' or 'enhanced'

        Returns:
            Dict containing trained model and metadata
        """
        model_name = model_config['name']
        self.logger.info(f"Training {model_name} ({model_type} features)")

        # Get model instance
        model = model_config['model']

        # Train model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        # Cross-validation score
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc'
        )

        # Create result dictionary
        result = {
            'model': model,
            'model_name': model_name,
            'model_type': model_type,
            'training_time_seconds': training_time,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_names': list(X_train.columns),
            'feature_count': len(X_train.columns),
            'training_samples': len(X_train)
        }

        # Add feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            result['feature_importance'] = feature_importance
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(X_train.columns, abs(model.coef_[0])))
            result['feature_importance'] = feature_importance

        self.logger.info(f"{model_name} training completed. CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return result

    def train_baseline_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train all models on baseline features (4 features)

        Args:
            X_train: Baseline training features
            y_train: Training target

        Returns:
            Dict containing all trained baseline models
        """
        self.logger.info("Starting baseline model training")

        baseline_results = {}

        for model_key, model_config in self.model_configs.items():
            try:
                result = self.train_single_model(model_config, X_train, y_train, 'baseline')
                baseline_results[model_key] = result
            except Exception as e:
                self.logger.error(f"Error training baseline {model_config['name']}: {str(e)}")

        self.logger.info(f"Baseline training completed for {len(baseline_results)} models")
        return baseline_results

    def train_enhanced_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train all models on enhanced features (10+ features)

        Args:
            X_train: Enhanced training features
            y_train: Training target

        Returns:
            Dict containing all trained enhanced models
        """
        self.logger.info("Starting enhanced model training")

        enhanced_results = {}

        for model_key, model_config in self.model_configs.items():
            try:
                result = self.train_single_model(model_config, X_train, y_train, 'enhanced')
                enhanced_results[model_key] = result
            except Exception as e:
                self.logger.error(f"Error training enhanced {model_config['name']}: {str(e)}")

        self.logger.info(f"Enhanced training completed for {len(enhanced_results)} models")
        return enhanced_results

    def save_trained_models(self, baseline_results: Dict[str, Any],
                            enhanced_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save all trained models and metadata

        Args:
            baseline_results: Baseline model results
            enhanced_results: Enhanced model results

        Returns:
            Dict containing paths to saved files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = {}

        self.logger.info("Saving trained models and metadata")

        # Save baseline models
        for model_key, result in baseline_results.items():
            model_file = self.artifacts_dir / f'step4_{model_key}_baseline_{timestamp}.pkl'

            # Save model
            with open(model_file, 'wb') as f:
                pickle.dump(result['model'], f)

            saved_files[f'{model_key}_baseline'] = str(model_file)

            # Save metadata (without the actual model object)
            metadata = {k: v for k, v in result.items() if k != 'model'}
            metadata_file = self.metadata_dir / f'step4_{model_key}_baseline_metadata_{timestamp}.json'

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

        # Save enhanced models
        for model_key, result in enhanced_results.items():
            model_file = self.artifacts_dir / f'step4_{model_key}_enhanced_{timestamp}.pkl'

            # Save model
            with open(model_file, 'wb') as f:
                pickle.dump(result['model'], f)

            saved_files[f'{model_key}_enhanced'] = str(model_file)

            # Save metadata
            metadata = {k: v for k, v in result.items() if k != 'model'}
            metadata_file = self.metadata_dir / f'step4_{model_key}_enhanced_metadata_{timestamp}.json'

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

        # Save combined training summary
        training_summary = {
            'timestamp': timestamp,
            'baseline_models': {k: {
                'cv_mean': v['cv_mean'],
                'cv_std': v['cv_std'],
                'feature_count': v['feature_count'],
                'training_time': v['training_time_seconds']
            } for k, v in baseline_results.items()},
            'enhanced_models': {k: {
                'cv_mean': v['cv_mean'],
                'cv_std': v['cv_std'],
                'feature_count': v['feature_count'],
                'training_time': v['training_time_seconds']
            } for k, v in enhanced_results.items()},
            'available_models': list(self.model_configs.keys()),
            'saved_files': saved_files
        }

        summary_file = self.metadata_dir / f'step4_training_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)

        saved_files['training_summary'] = str(summary_file)

        self.logger.info(f"All models and metadata saved with timestamp: {timestamp}")
        return saved_files

    def train_all_models(self, data_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
        """
        Main training pipeline for both baseline and enhanced models

        Args:
            data_path: Path to processed data directory

        Returns:
            Tuple containing baseline results, enhanced results, and saved file paths
        """
        self.logger.info("Starting complete model training pipeline")

        try:
            # Load datasets
            datasets = self.load_training_data(data_path)

            # Prepare baseline features and target
            X_baseline_train, y_baseline_train = self.prepare_features_target(datasets['baseline_train'])

            # Prepare enhanced features and target
            X_enhanced_train, y_enhanced_train = self.prepare_features_target(datasets['enhanced_train'])

            # Train baseline models
            baseline_results = self.train_baseline_models(X_baseline_train, y_baseline_train)

            # Train enhanced models
            enhanced_results = self.train_enhanced_models(X_enhanced_train, y_enhanced_train)

            # Save all models
            saved_files = self.save_trained_models(baseline_results, enhanced_results)

            # Log summary
            self.logger.info("Model training pipeline completed successfully")
            self.logger.info(f"Baseline models trained: {len(baseline_results)}")
            self.logger.info(f"Enhanced models trained: {len(enhanced_results)}")

            # Log best performing models
            if baseline_results:
                best_baseline = max(baseline_results.items(), key=lambda x: x[1]['cv_mean'])
                self.logger.info(f"Best baseline model: {best_baseline[0]} (AUC: {best_baseline[1]['cv_mean']:.4f})")

            if enhanced_results:
                best_enhanced = max(enhanced_results.items(), key=lambda x: x[1]['cv_mean'])
                self.logger.info(f"Best enhanced model: {best_enhanced[0]} (AUC: {best_enhanced[1]['cv_mean']:.4f})")

            return baseline_results, enhanced_results, saved_files

        except Exception as e:
            self.logger.error(f"Error in training pipeline: {str(e)}")
            raise