"""
Data Loader Module for NBE Prediction Project
Handles loading and initial validation of the icuc_ml_dataset.xlsx
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import json


class DataLoader:
    """
    Handles loading and initial validation of the NBE dataset
    """

    def __init__(self, data_path: Path, log_path: Path):
        self.data_path = data_path
        self.log_path = log_path
        self.logger = self._setup_logger()

        # Expected schema for validation
        self.expected_columns = {
            'accident_number': 'object',  # Patient identifier
            'p_score': 'int64',  # Pain score (0-4)
            'p_status': 'int64',  # Pain status (0-2)
            'fl_score': 'int64',  # Function limitation score (0-4)
            'fl_status': 'int64',  # Function limitation status (0-2)
            'nbe': 'int64'  # Target variable (0, 1, 2)
        }

        # Valid ranges for validation
        self.valid_ranges = {
            'p_score': (0, 4),
            'p_status': (0, 2),
            'fl_score': (0, 4),
            'fl_status': (0, 2),
            'nbe': (0, 2)
        }

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for data loading operations"""
        logger = logging.getLogger('DataLoader')
        logger.setLevel(logging.INFO)

        # Create log directory if it doesn't exist
        log_dir = self.log_path / 'step1'
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'data_loader_{timestamp}.log'

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler to logger
        if not logger.handlers:
            logger.addHandler(file_handler)

        return logger

    def load_excel_data(self, filename: str = 'icuc_ml_dataset.xlsx') -> pd.DataFrame:
        """
        Load data from Excel file with error handling

        Args:
            filename: Name of the Excel file to load

        Returns:
            pd.DataFrame: Loaded dataset

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path = self.data_path / 'raw' / filename

        self.logger.info(f"Starting data loading from: {file_path}")

        try:
            # Check if file exists
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")

            # Load Excel file
            df = pd.read_excel(file_path)

            self.logger.info(f"Successfully loaded data with shape: {df.shape}")
            self.logger.info(f"Columns found: {list(df.columns)}")

            return df

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def validate_data_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the loaded data against expected schema

        Args:
            df: DataFrame to validate

        Returns:
            Dict containing validation results
        """
        self.logger.info("Starting data schema validation")

        validation_results = {
            'is_valid': True,
            'missing_columns': [],
            'extra_columns': [],
            'type_mismatches': [],
            'range_violations': {},
            'summary': {}
        }

        # Check for missing columns
        missing_cols = set(self.expected_columns.keys()) - set(df.columns)
        if missing_cols:
            validation_results['missing_columns'] = list(missing_cols)
            validation_results['is_valid'] = False
            self.logger.warning(f"Missing columns: {missing_cols}")

        # Check for extra columns
        extra_cols = set(df.columns) - set(self.expected_columns.keys())
        if extra_cols:
            validation_results['extra_columns'] = list(extra_cols)
            self.logger.info(f"Extra columns found: {extra_cols}")

        # Validate data types and ranges for expected columns
        for col in self.expected_columns.keys():
            if col in df.columns:
                # Check data types (skip for now, focus on ranges)

                # Check valid ranges for numeric columns
                if col in self.valid_ranges:
                    min_val, max_val = self.valid_ranges[col]
                    invalid_values = df[~df[col].between(min_val, max_val, na_action='ignore')][col]

                    if not invalid_values.empty:
                        validation_results['range_violations'][col] = {
                            'expected_range': f"{min_val}-{max_val}",
                            'invalid_count': len(invalid_values),
                            'invalid_values': invalid_values.unique().tolist()
                        }
                        validation_results['is_valid'] = False
                        self.logger.warning(f"Range violations in {col}: {len(invalid_values)} values")

        # Generate summary statistics
        validation_results['summary'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'duplicate_rows': df.duplicated().sum(),
            'unique_patients': df['accident_number'].nunique() if 'accident_number' in df.columns else None
        }

        self.logger.info(f"Schema validation completed. Valid: {validation_results['is_valid']}")
        return validation_results

    def log_data_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate and log comprehensive data quality metrics

        Args:
            df: DataFrame to analyze

        Returns:
            Dict containing data quality metrics
        """
        self.logger.info("Calculating data quality metrics")

        quality_metrics = {}

        # Basic statistics
        quality_metrics['basic_stats'] = {
            'shape': df.shape,
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': round(df.duplicated().sum() / len(df) * 100, 2)
        }

        # Missing values analysis
        missing_stats = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_stats[col] = {
                'count': missing_count,
                'percentage': round(missing_count / len(df) * 100, 2)
            }
        quality_metrics['missing_values'] = missing_stats

        # Feature-specific analysis
        feature_stats = {}
        for col in ['p_score', 'p_status', 'fl_score', 'fl_status', 'nbe']:
            if col in df.columns:
                feature_stats[col] = {
                    'unique_values': df[col].nunique(),
                    'value_counts': df[col].value_counts().to_dict(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': round(df[col].mean(), 2),
                    'std': round(df[col].std(), 2)
                }
        quality_metrics['feature_analysis'] = feature_stats

        # Target variable analysis
        if 'nbe' in df.columns:
            target_analysis = {
                'class_distribution': df['nbe'].value_counts().to_dict(),
                'class_percentages': (df['nbe'].value_counts(normalize=True) * 100).round(2).to_dict(),
                'is_balanced': self._check_class_balance(df['nbe'])
            }
            quality_metrics['target_analysis'] = target_analysis

        # Patient-level analysis
        if 'accident_number' in df.columns:
            patient_stats = df.groupby('accident_number').size()
            quality_metrics['patient_analysis'] = {
                'unique_patients': df['accident_number'].nunique(),
                'consultations_per_patient': {
                    'mean': round(patient_stats.mean(), 2),
                    'median': patient_stats.median(),
                    'min': patient_stats.min(),
                    'max': patient_stats.max(),
                    'std': round(patient_stats.std(), 2)
                }
            }

        # Log key metrics
        self.logger.info(f"Data shape: {quality_metrics['basic_stats']['shape']}")
        self.logger.info(
            f"Unique patients: {quality_metrics.get('patient_analysis', {}).get('unique_patients', 'N/A')}")
        self.logger.info(f"Missing values: {sum([v['count'] for v in missing_stats.values()])} total")

        if 'target_analysis' in quality_metrics:
            self.logger.info(f"Target distribution: {quality_metrics['target_analysis']['class_distribution']}")

        return quality_metrics

    def _check_class_balance(self, target_series: pd.Series, threshold: float = 0.3) -> bool:
        """
        Check if target classes are reasonably balanced

        Args:
            target_series: Target variable series
            threshold: Minimum proportion for smallest class

        Returns:
            bool: True if classes are reasonably balanced
        """
        class_proportions = target_series.value_counts(normalize=True)
        min_proportion = class_proportions.min()
        return min_proportion >= threshold

    def save_exploration_results(self, df: pd.DataFrame, validation_results: Dict,
                                 quality_metrics: Dict) -> Path:
        """
        Save exploration results to processed data folder

        Args:
            df: Original dataframe
            validation_results: Schema validation results
            quality_metrics: Data quality metrics

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save processed data
        processed_dir = self.data_path / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)

        output_file = processed_dir / f'step1_data_exploration_{timestamp}.csv'
        df.to_csv(output_file, index=False)

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'validation_results': validation_results,
            'quality_metrics': quality_metrics,
            'file_path': str(output_file)
        }

        metadata_file = processed_dir / f'step1_metadata_{timestamp}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.info(f"Exploration results saved to: {output_file}")
        self.logger.info(f"Metadata saved to: {metadata_file}")

        return output_file