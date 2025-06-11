"""
Data Cleaner Module for NBE Prediction Project
Handles data cleaning and preparation for anonymization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
from datetime import datetime
import json


class DataCleaner:
    """
    Handles data cleaning and preparation for machine learning pipeline
    """

    def __init__(self, data_path: Path, log_path: Path):
        self.data_path = data_path
        self.log_path = log_path
        self.logger = self._setup_logger()

        # Business rules for cleaning
        self.feature_ranges = {
            'p_score': (0, 4),
            'p_status': (0, 2),
            'fl_score': (0, 4),
            'fl_status': (0, 2),
            'nbe': (0, 2)
        }

        # Columns required for processing
        self.required_columns = [
            'accident_number', 'accident_date', 'contact_date',
            'p_score', 'p_status', 'fl_score', 'fl_status', 'nbe'
        ]

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for data cleaning operations"""
        logger = logging.getLogger('DataCleaner')
        logger.setLevel(logging.INFO)

        # Create log directory
        log_dir = self.log_path / 'step2'
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'data_cleaner_{timestamp}.log'

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(file_handler)

        return logger

    def load_step1_data(self, filename: str = None) -> pd.DataFrame:
        """
        Load the most recent Step 1 processed data

        Args:
            filename: Optional specific filename to load

        Returns:
            pd.DataFrame: Loaded dataset from Step 1
        """
        self.logger.info("Loading Step 1 processed data")

        processed_dir = self.data_path / 'processed'
        self.logger.info(f"Looking for Step 1 files in: {processed_dir}")

        if filename:
            file_path = processed_dir / filename
            self.logger.info(f"Using specified filename: {filename}")
        else:
            # Find the most recent step1 file
            step1_files = list(processed_dir.glob('step1_data_exploration_*.csv'))
            self.logger.info(f"Found Step 1 files: {step1_files}")

            if not step1_files:
                # List all files in processed directory for debugging
                all_files = list(processed_dir.glob('*'))
                self.logger.error(f"No Step 1 files found. All files in {processed_dir}: {all_files}")
                raise FileNotFoundError(f"No Step 1 processed files found in {processed_dir}")

            file_path = max(step1_files, key=lambda x: x.stat().st_mtime)

        self.logger.info(f"Loading data from: {file_path}")

        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File does not exist: {file_path}")

            df = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded {len(df)} records from Step 1")
            return df

        except Exception as e:
            self.logger.error(f"Error loading Step 1 data: {str(e)}")
            raise

    def validate_required_columns(self, df: pd.DataFrame) -> bool:
        """
        Validate that all required columns are present

        Args:
            df: DataFrame to validate

        Returns:
            bool: True if all required columns present
        """
        missing_cols = set(self.required_columns) - set(df.columns)

        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False

        self.logger.info("All required columns present")
        return True

    def clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize data types

        Args:
            df: DataFrame to clean

        Returns:
            pd.DataFrame: DataFrame with cleaned data types
        """
        self.logger.info("Cleaning data types")

        df_clean = df.copy()

        # Convert date columns
        date_columns = ['accident_date', 'contact_date']
        for col in date_columns:
            if col in df_clean.columns:
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col])
                    self.logger.info(f"Converted {col} to datetime")
                except Exception as e:
                    self.logger.warning(f"Could not convert {col} to datetime: {str(e)}")

        # Convert numeric columns
        numeric_columns = ['p_score', 'p_status', 'fl_score', 'fl_status', 'nbe']
        for col in numeric_columns:
            if col in df_clean.columns:
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    self.logger.info(f"Converted {col} to numeric")
                except Exception as e:
                    self.logger.warning(f"Could not convert {col} to numeric: {str(e)}")

        return df_clean

    def remove_nbe_no_info_cases(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Remove cases where NBE = 2 (No Information) for binary classification

        Args:
            df: DataFrame to filter

        Returns:
            Tuple[pd.DataFrame, Dict]: Filtered dataframe and removal statistics
        """
        self.logger.info("Removing NBE 'No Information' cases (NBE=2)")

        original_count = len(df)
        original_distribution = df['nbe'].value_counts().to_dict()

        # Filter out NBE = 2 cases
        df_binary = df[df['nbe'].isin([0, 1])].copy()

        removed_count = original_count - len(df_binary)
        final_distribution = df_binary['nbe'].value_counts().to_dict()

        # Calculate statistics
        removal_stats = {
            'original_records': original_count,
            'final_records': len(df_binary),
            'removed_records': removed_count,
            'removal_percentage': (removed_count / original_count) * 100,
            'original_distribution': original_distribution,
            'final_distribution': final_distribution,
            'final_class_balance': {
                'nbe_no_percentage': (final_distribution.get(0, 0) / len(df_binary)) * 100,
                'nbe_yes_percentage': (final_distribution.get(1, 0) / len(df_binary)) * 100
            }
        }

        self.logger.info(f"Removed {removed_count} records ({removal_stats['removal_percentage']:.1f}%)")
        self.logger.info(f"Final dataset: {len(df_binary)} records")
        self.logger.info(f"Final class distribution: {final_distribution}")

        return df_binary, removal_stats

    def validate_feature_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that all features are within expected ranges

        Args:
            df: DataFrame to validate

        Returns:
            Dict: Validation results
        """
        self.logger.info("Validating feature ranges")

        validation_results = {
            'is_valid': True,
            'range_violations': {},
            'summary': {}
        }

        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature in df.columns:
                # Check for values outside valid range
                valid_mask = df[feature].between(min_val, max_val) | df[feature].isnull()
                violations = df[~valid_mask]

                if len(violations) > 0:
                    validation_results['range_violations'][feature] = {
                        'count': len(violations),
                        'percentage': (len(violations) / len(df)) * 100,
                        'invalid_values': violations[feature].unique().tolist()
                    }
                    validation_results['is_valid'] = False
                    self.logger.warning(f"Range violations in {feature}: {len(violations)} cases")
                else:
                    self.logger.info(f"{feature}: All values within valid range {min_val}-{max_val}")

        validation_results['summary'] = {
            'total_features_checked': len([f for f in self.feature_ranges.keys() if f in df.columns]),
            'features_with_violations': len(validation_results['range_violations']),
            'overall_violation_rate': sum([v['count'] for v in validation_results['range_violations'].values()]) / len(df) * 100 if len(df) > 0 else 0
        }

        return validation_results

    def handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handle missing values in the dataset

        Args:
            df: DataFrame to process

        Returns:
            Tuple[pd.DataFrame, Dict]: Processed dataframe and missing value statistics
        """
        self.logger.info("Analyzing missing values")

        missing_stats = {}
        df_clean = df.copy()

        # Analyze missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_percentage = (missing_count / len(df)) * 100
                missing_stats[col] = {
                    'count': missing_count,
                    'percentage': missing_percentage
                }
                self.logger.warning(f"{col}: {missing_count} missing values ({missing_percentage:.2f}%)")

        # Handle missing values based on column type
        if missing_stats:
            self.logger.info("Handling missing values")

            # For critical features, we might need to remove records
            critical_features = ['p_score', 'p_status', 'fl_score', 'fl_status', 'nbe']

            for feature in critical_features:
                if feature in missing_stats:
                    missing_count = missing_stats[feature]['count']
                    self.logger.warning(f"Critical feature {feature} has {missing_count} missing values")

                    # For now, remove records with missing critical features
                    df_clean = df_clean.dropna(subset=[feature])
                    self.logger.info(f"Removed records with missing {feature}")

        else:
            self.logger.info("No missing values found")

        final_missing_stats = {
            'original_missing_analysis': missing_stats,
            'records_removed_due_to_missing': len(df) - len(df_clean),
            'final_missing_count': df_clean.isnull().sum().sum()
        }

        return df_clean, final_missing_stats

    def detect_and_handle_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect and handle duplicate records

        Args:
            df: DataFrame to process

        Returns:
            Tuple[pd.DataFrame, Dict]: Processed dataframe and duplicate statistics
        """
        self.logger.info("Detecting duplicate records")

        original_count = len(df)

        # Check for exact duplicates
        exact_duplicates = df.duplicated().sum()

        # Check for potential duplicates (same patient, same date)
        if 'accident_number' in df.columns and 'contact_date' in df.columns:
            patient_date_duplicates = df.duplicated(subset=['accident_number', 'contact_date']).sum()
        else:
            patient_date_duplicates = 0

        duplicate_stats = {
            'original_records': original_count,
            'exact_duplicates': exact_duplicates,
            'patient_date_duplicates': patient_date_duplicates
        }

        # Remove exact duplicates if found
        df_clean = df.drop_duplicates()
        final_count = len(df_clean)

        duplicate_stats.update({
            'final_records': final_count,
            'records_removed': original_count - final_count
        })

        if duplicate_stats['records_removed'] > 0:
            self.logger.warning(f"Removed {duplicate_stats['records_removed']} duplicate records")
        else:
            self.logger.info("No duplicate records found")

        return df_clean, duplicate_stats

    def generate_cleaning_summary(self, df_original: pd.DataFrame, df_final: pd.DataFrame,
                                removal_stats: Dict, validation_results: Dict,
                                missing_stats: Dict, duplicate_stats: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive cleaning summary

        Args:
            df_original: Original dataframe
            df_final: Final cleaned dataframe
            removal_stats: NBE removal statistics
            validation_results: Feature validation results
            missing_stats: Missing value statistics
            duplicate_stats: Duplicate detection statistics

        Returns:
            Dict: Comprehensive cleaning summary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_flow': {
                'original_records': len(df_original),
                'after_nbe_filtering': removal_stats['final_records'],
                'after_missing_value_handling': len(df_final) + missing_stats['records_removed_due_to_missing'],
                'after_duplicate_removal': len(df_final),
                'final_records': len(df_final),
                'total_records_removed': len(df_original) - len(df_final),
                'data_retention_rate': (len(df_final) / len(df_original)) * 100
            },
            'nbe_filtering': removal_stats,
            'feature_validation': validation_results,
            'missing_value_handling': missing_stats,
            'duplicate_detection': duplicate_stats,
            'final_data_quality': {
                'shape': df_final.shape,
                'missing_values': df_final.isnull().sum().sum(),
                'unique_patients': df_final['accident_number'].nunique() if 'accident_number' in df_final.columns else None,
                'date_range': {
                    'earliest_accident': df_final['accident_date'].min().strftime('%Y-%m-%d') if 'accident_date' in df_final.columns else None,
                    'latest_contact': df_final['contact_date'].max().strftime('%Y-%m-%d') if 'contact_date' in df_final.columns else None
                } if 'accident_date' in df_final.columns and 'contact_date' in df_final.columns else None
            }
        }

        return summary

    def clean_dataset(self, input_filename: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main cleaning pipeline

        Args:
            input_filename: Optional specific input file

        Returns:
            Tuple[pd.DataFrame, Dict]: Cleaned dataframe and comprehensive statistics
        """
        self.logger.info("Starting data cleaning pipeline")

        try:
            # Step 1: Load data
            df = self.load_step1_data(input_filename)
            df_original = df.copy()

            # Step 2: Validate schema
            if not self.validate_required_columns(df):
                raise ValueError("Required columns missing")

            # Step 3: Clean data types
            df = self.clean_data_types(df)

            # Step 4: Remove NBE "No Information" cases for binary classification
            df, removal_stats = self.remove_nbe_no_info_cases(df)

            # Step 5: Validate feature ranges
            validation_results = self.validate_feature_ranges(df)

            # Step 6: Handle missing values
            df, missing_stats = self.handle_missing_values(df)

            # Step 7: Handle duplicates
            df, duplicate_stats = self.detect_and_handle_duplicates(df)

            # Step 8: Generate summary
            cleaning_summary = self.generate_cleaning_summary(
                df_original, df, removal_stats, validation_results,
                missing_stats, duplicate_stats
            )

            self.logger.info(f"Data cleaning completed successfully")
            self.logger.info(f"Final dataset: {len(df)} records from {df['accident_number'].nunique()} patients")

            return df, cleaning_summary

        except Exception as e:
            self.logger.error(f"Error in data cleaning pipeline: {str(e)}")
            raise

    def save_cleaned_data(self, df: pd.DataFrame, cleaning_summary: Dict[str, Any]) -> Path:
        """
        Save cleaned data and metadata

        Args:
            df: Cleaned dataframe
            cleaning_summary: Cleaning summary statistics

        Returns:
            Path: Path to saved cleaned data file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create processed directory
        processed_dir = self.data_path / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Save cleaned data
        output_file = processed_dir / f'step2_cleaned_data_{timestamp}.csv'
        df.to_csv(output_file, index=False)

        # Save cleaning summary
        summary_file = processed_dir / f'step2_cleaning_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(cleaning_summary, f, indent=2, default=str)

        self.logger.info(f"Cleaned data saved to: {output_file}")
        self.logger.info(f"Cleaning summary saved to: {summary_file}")

        return output_file