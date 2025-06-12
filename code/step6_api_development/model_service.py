"""
Model Service Module for NBE Prediction Project
Handles model loading, feature engineering, and prediction logic
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
from datetime import datetime
import json
import glob

logger = logging.getLogger(__name__)


class ModelService:
    """
    Service class for managing model loading and predictions
    """

    def __init__(self, models_path: Path):
        self.models_path = models_path
        self.baseline_model = None
        self.enhanced_model = None
        self.baseline_metadata = None
        self.enhanced_metadata = None
        self.models_loaded = False

        # Feature definitions (must match training)
        self.baseline_features = ['p_score', 'p_status', 'fl_score', 'fl_status']
        self.enhanced_features = [
            'p_score', 'p_status', 'fl_score', 'fl_status',
            'days_since_accident', 'consultation_number',
            'p_score_fl_score_interaction', 'severity_index',
            'both_improving', 'high_severity'
        ]

    def load_models(self) -> bool:
        """
        Load both baseline and enhanced models from artifacts

        Returns:
            bool: True if models loaded successfully
        """
        try:
            logger.info("Loading NBE prediction models...")

            # Find model files
            artifacts_dir = self.models_path / 'artifacts'
            metadata_dir = self.models_path / 'metadata'

            if not artifacts_dir.exists() or not metadata_dir.exists():
                logger.error(f"Model directories not found: {artifacts_dir}, {metadata_dir}")
                return False

            # Load baseline model (Logistic Regression)
            baseline_files = list(artifacts_dir.glob('step4_logistic_regression_baseline_*.pkl'))
            if not baseline_files:
                logger.error("Baseline model file not found")
                return False

            baseline_model_file = max(baseline_files, key=lambda x: x.stat().st_mtime)
            with open(baseline_model_file, 'rb') as f:
                self.baseline_model = pickle.load(f)

            # Load baseline metadata
            baseline_metadata_files = list(metadata_dir.glob('step4_logistic_regression_baseline_metadata_*.json'))
            if baseline_metadata_files:
                baseline_metadata_file = max(baseline_metadata_files, key=lambda x: x.stat().st_mtime)
                with open(baseline_metadata_file, 'r') as f:
                    self.baseline_metadata = json.load(f)

            # Load enhanced model (XGBoost)
            enhanced_files = list(artifacts_dir.glob('step4_xgboost_enhanced_*.pkl'))
            if not enhanced_files:
                logger.error("Enhanced model file not found")
                return False

            enhanced_model_file = max(enhanced_files, key=lambda x: x.stat().st_mtime)
            with open(enhanced_model_file, 'rb') as f:
                self.enhanced_model = pickle.load(f)

            # Load enhanced metadata
            enhanced_metadata_files = list(metadata_dir.glob('step4_xgboost_enhanced_metadata_*.json'))
            if enhanced_metadata_files:
                enhanced_metadata_file = max(enhanced_metadata_files, key=lambda x: x.stat().st_mtime)
                with open(enhanced_metadata_file, 'r') as f:
                    self.enhanced_metadata = json.load(f)

            self.models_loaded = True
            logger.info("✅ Models loaded successfully")
            logger.info(f"   Baseline: {baseline_model_file.name}")
            logger.info(f"   Enhanced: {enhanced_model_file.name}")

            return True

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction and derived features (same as training)

        Args:
            df: DataFrame with core features

        Returns:
            pd.DataFrame: DataFrame with interaction features added
        """
        df_interact = df.copy()

        # Pain and function limitation interaction
        df_interact['p_score_fl_score_interaction'] = df_interact['p_score'] * df_interact['fl_score']

        # Severity index (combined score)
        df_interact['severity_index'] = (df_interact['p_score'] + df_interact['fl_score']) / 2

        # Combined improvement indicator (both status = 2)
        df_interact['both_improving'] = ((df_interact['p_status'] == 2) & (df_interact['fl_status'] == 2)).astype(int)

        # High severity indicator (both scores >= 3)
        df_interact['high_severity'] = ((df_interact['p_score'] >= 3) & (df_interact['fl_score'] >= 3)).astype(int)

        return df_interact

    def prepare_baseline_features(self, request_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare baseline features for prediction

        Args:
            request_data: Request data dictionary

        Returns:
            pd.DataFrame: Prepared features for baseline model
        """
        # Extract core features
        features_data = {
            'p_score': [request_data['p_score']],
            'p_status': [request_data['p_status']],
            'fl_score': [request_data['fl_score']],
            'fl_status': [request_data['fl_status']]
        }

        df = pd.DataFrame(features_data)

        # Ensure correct column order
        df = df[self.baseline_features]

        logger.debug(f"Baseline features prepared: {df.shape}, columns: {list(df.columns)}")
        return df

    def prepare_enhanced_features(self, request_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare enhanced features for prediction

        Args:
            request_data: Request data dictionary

        Returns:
            pd.DataFrame: Prepared features for enhanced model
        """
        # Start with core features
        features_data = {
            'p_score': [request_data['p_score']],
            'p_status': [request_data['p_status']],
            'fl_score': [request_data['fl_score']],
            'fl_status': [request_data['fl_status']],
            'days_since_accident': [request_data.get('days_since_accident', 21)],
            'consultation_number': [request_data.get('consultation_number', 2)]
        }

        df = pd.DataFrame(features_data)

        # Create interaction features
        df = self.create_interaction_features(df)

        # Ensure correct column order (same as training)
        df = df[self.enhanced_features]

        logger.debug(f"Enhanced features prepared: {df.shape}, columns: {list(df.columns)}")
        return df

    def calculate_confidence_level(self, probability: float) -> str:
        """
        Calculate confidence level based on prediction probability

        Args:
            probability: Prediction probability

        Returns:
            str: Confidence level (low, medium, high)
        """
        # Convert to distance from 0.5 (uncertainty)
        confidence_score = abs(probability - 0.5) * 2

        if confidence_score >= 0.7:
            return "high"
        elif confidence_score >= 0.4:
            return "medium"
        else:
            return "low"

    def predict_baseline(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction using baseline model

        Args:
            request_data: Request data dictionary

        Returns:
            Dict: Prediction results
        """
        if not self.models_loaded or self.baseline_model is None:
            raise ValueError("Baseline model not loaded")

        try:
            # Prepare features
            X = self.prepare_baseline_features(request_data)

            # Make prediction
            prediction_proba = self.baseline_model.predict_proba(X)[0]
            nbe_no_prob = float(prediction_proba[0])
            nbe_yes_prob = float(prediction_proba[1])

            # Calculate confidence
            confidence = self.calculate_confidence_level(nbe_yes_prob)

            result = {
                'nbe_yes_probability': nbe_yes_prob,
                'nbe_no_probability': nbe_no_prob,
                'confidence_level': confidence,
                'model_used': 'logistic_regression',
                'model_type': 'baseline',
                'prediction_timestamp': datetime.now(),
                'feature_engineering_applied': False,
                'input_validation_passed': True
            }

            logger.info(f"Baseline prediction: NBE Yes: {nbe_yes_prob:.3f}, Confidence: {confidence}")
            return result

        except Exception as e:
            logger.error(f"Error in baseline prediction: {str(e)}")
            raise

    def predict_enhanced(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction using enhanced model

        Args:
            request_data: Request data dictionary

        Returns:
            Dict: Prediction results
        """
        if not self.models_loaded or self.enhanced_model is None:
            raise ValueError("Enhanced model not loaded")

        try:
            # Prepare features
            X = self.prepare_enhanced_features(request_data)

            # Make prediction
            prediction_proba = self.enhanced_model.predict_proba(X)[0]
            nbe_no_prob = float(prediction_proba[0])
            nbe_yes_prob = float(prediction_proba[1])

            # Calculate confidence
            confidence = self.calculate_confidence_level(nbe_yes_prob)

            result = {
                'nbe_yes_probability': nbe_yes_prob,
                'nbe_no_probability': nbe_no_prob,
                'confidence_level': confidence,
                'model_used': 'xgboost',
                'model_type': 'enhanced',
                'prediction_timestamp': datetime.now(),
                'feature_engineering_applied': True,
                'input_validation_passed': True
            }

            logger.info(f"Enhanced prediction: NBE Yes: {nbe_yes_prob:.3f}, Confidence: {confidence}")
            return result

        except Exception as e:
            logger.error(f"Error in enhanced prediction: {str(e)}")
            raise

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the model service

        Returns:
            Dict: Health status information
        """
        models_status = {
            'baseline_model': 'loaded' if self.baseline_model is not None else 'not_loaded',
            'enhanced_model': 'loaded' if self.enhanced_model is not None else 'not_loaded'
        }

        overall_status = 'healthy' if self.models_loaded else 'unhealthy'

        return {
            'status': overall_status,
            'timestamp': datetime.now(),
            'models_loaded': models_status,
            'api_version': '1.0.0'
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about loaded models

        Returns:
            Dict: Model information
        """
        baseline_info = {
            'algorithm': 'LogisticRegression',
            'features': len(self.baseline_features),
            'auc_score': self.baseline_metadata.get('cv_mean', 0.746) if self.baseline_metadata else 0.746,
            'training_samples': self.baseline_metadata.get('training_samples', 4365) if self.baseline_metadata else 4365
        }

        enhanced_info = {
            'algorithm': 'XGBoost',
            'features': len(self.enhanced_features),
            'auc_score': self.enhanced_metadata.get('cv_mean', 0.801) if self.enhanced_metadata else 0.801,
            'training_samples': self.enhanced_metadata.get('training_samples', 4365) if self.enhanced_metadata else 4365
        }

        training_timestamp = "2025-06-12T11:27:36"
        if self.baseline_metadata and 'timestamp' in self.baseline_metadata:
            training_timestamp = self.baseline_metadata['timestamp']

        return {
            'baseline_model': baseline_info,
            'enhanced_model': enhanced_info,
            'training_timestamp': training_timestamp,
            'feature_sets': {
                'baseline': self.baseline_features,
                'enhanced': self.enhanced_features
            }
        }