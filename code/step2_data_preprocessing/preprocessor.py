"""
Preprocessor Module for NBE Prediction Project
Handles feature engineering and train/test splitting for both baseline and enhanced models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
from datetime import datetime
import json
from sklearn.model_selection import train_test_split


class DualFeaturePreprocessor:
    """
    Handles feature engineering for both baseline (4-feature) and enhanced (6-feature) models
    """

    def __init__(self, data_path: Path, log_path: Path):
        self.data_path = data_path
        self.log_path = log_path
        self.logger = self._setup_logger()

        # Feature configurations
        self.baseline_features = ['p_score', 'p_status', 'fl_score', 'fl_status']
        self.enhanced_features = [
            'p_score', 'p_status', 'fl_score', 'fl_status',
            'days_since_accident', 'consultation_number'
        ]
        self.target_column = 'nbe'

        # Random state for reproducibility
        self.random_state = 42

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for preprocessing operations"""
        logger = logging.getLogger('DualFeaturePreprocessor')
        logger.setLevel(logging.INFO)

        # Create log directory
        log_dir = self.log_path / 'step2'
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'preprocessor_{timestamp}.log'

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(file_handler)

        return logger

    def load_anonymized_data(self, input_filename: str = None) -> pd.DataFrame:
        """
        Load anonymized data from Step 2

        Args:
            input_filename: Optional specific filename

        Returns:
            pd.DataFrame: Loaded anonymized dataset
        """
        self.logger.info("Loading anonymized data from Step 2")

        processed_dir = self.data_path / 'processed'

        if input_filename:
            file_path = processed_dir / input_filename
        else:
            # Find most recent anonymized data file
            anon_files = list(processed_dir.glob('step2_anonymized_data_*.csv'))
            if not anon_files:
                raise FileNotFoundError("No Step 2 anonymized data files found")
            file_path = max(anon_files, key=lambda x: x.stat().st_mtime)

        self.logger.info(f"Loading data from: {file_path}")

        try:
            df = pd.read_csv(file_path)

            # Convert date columns if present
            date_columns = ['accident_date', 'contact_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            self.logger.info(f"Successfully loaded {len(df)} records from {df['anonymous_patient_id'].nunique()} patients")
            return df

        except Exception as e:
            self.logger.error(f"Error loading anonymized data: {str(e)}")
            raise

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from date columns

        Args:
            df: DataFrame with date columns

        Returns:
            pd.DataFrame: DataFrame with temporal features added
        """
        self.logger.info("Creating temporal features")

        df_temp = df.copy()

        # Create days_since_accident feature
        if 'accident_date' in df.columns and 'contact_date' in df.columns:
            df_temp['days_since_accident'] = (df_temp['contact_date'] - df_temp['accident_date']).dt.days

            # Validate temporal feature
            invalid_days = df_temp['days_since_accident'] < 0
            if invalid_days.any():
                self.logger.warning(f"Found {invalid_days.sum()} records with negative days_since_accident")
                # Set negative values to 0 (same day)
                df_temp.loc[invalid_days, 'days_since_accident'] = 0

            self.logger.info(f"Created days_since_accident feature (range: {df_temp['days_since_accident'].min()} to {df_temp['days_since_accident'].max()} days)")

        else:
            self.logger.warning("Date columns not found - cannot create days_since_accident feature")
            df_temp['days_since_accident'] = 0

        return df_temp

    def create_consultation_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create consultation sequence features

        Args:
            df: DataFrame with anonymous_patient_id

        Returns:
            pd.DataFrame: DataFrame with consultation sequence features added
        """
        self.logger.info("Creating consultation sequence features")

        df_seq = df.copy()

        # Sort by patient and contact date to establish sequence
        if 'contact_date' in df.columns:
            df_seq = df_seq.sort_values(['anonymous_patient_id', 'contact_date'])
        else:
            df_seq = df_seq.sort_values(['anonymous_patient_id'])

        # Create consultation number (1st, 2nd, 3rd, etc. for each patient)
        df_seq['consultation_number'] = df_seq.groupby('anonymous_patient_id').cumcount() + 1

        # Create additional sequence features
        df_seq['is_first_consultation'] = (df_seq['consultation_number'] == 1).astype(int)
        df_seq['is_follow_up'] = (df_seq['consultation_number'] > 1).astype(int)

        # Calculate consultations per patient
        consultations_per_patient = df_seq.groupby('anonymous_patient_id')['consultation_number'].max()
        df_seq['total_consultations_for_patient'] = df_seq['anonymous_patient_id'].map(consultations_per_patient)

        sequence_stats = {
            'max_consultation_number': df_seq['consultation_number'].max(),
            'patients_with_single_consultation': (consultations_per_patient == 1).sum(),
            'patients_with_multiple_consultations': (consultations_per_patient > 1).sum(),
            'mean_consultations_per_patient': consultations_per_patient.mean()
        }

        self.logger.info(f"Created consultation sequence features")
        self.logger.info(f"Consultation stats: {sequence_stats}")

        return df_seq

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction and derived features

        Args:
            df: DataFrame with core features

        Returns:
            pd.DataFrame: DataFrame with interaction features added
        """
        self.logger.info("Creating interaction and derived features")

        df_interact = df.copy()

        # Pain and function limitation interaction
        df_interact['p_score_fl_score_interaction'] = df_interact['p_score'] * df_interact['fl_score']

        # Severity index (combined score)
        df_interact['severity_index'] = (df_interact['p_score'] + df_interact['fl_score']) / 2

        # Status interaction (both improving/worsening)
        df_interact['p_status_fl_status_interaction'] = df_interact['p_status'] * df_interact['fl_status']

        # Combined improvement indicator (both status = 2)
        df_interact['both_improving'] = ((df_interact['p_status'] == 2) & (df_interact['fl_status'] == 2)).astype(int)

        # Combined worsening indicator (both status = 0)
        df_interact['both_worsening'] = ((df_interact['p_status'] == 0) & (df_interact['fl_status'] == 0)).astype(int)

        # High severity indicator (both scores >= 3)
        df_interact['high_severity'] = ((df_interact['p_score'] >= 3) & (df_interact['fl_score'] >= 3)).astype(int)

        # No symptoms indicator (both scores = 0)
        df_interact['no_symptoms'] = ((df_interact['p_score'] == 0) & (df_interact['fl_score'] == 0)).astype(int)

        interaction_stats = {
            'max_severity_index': df_interact['severity_index'].max(),
            'patients_both_improving': df_interact['both_improving'].sum(),
            'patients_both_worsening': df_interact['both_worsening'].sum(),
            'high_severity_cases': df_interact['high_severity'].sum(),
            'no_symptoms_cases': df_interact['no_symptoms'].sum()
        }

        self.logger.info(f"Created interaction features")
        self.logger.info(f"Interaction stats: {interaction_stats}")

        return df_interact

    def prepare_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataset with baseline features only

        Args:
            df: Full featured dataframe

        Returns:
            pd.DataFrame: DataFrame with baseline features + target
        """
        self.logger.info("Preparing baseline feature set")

        required_columns = self.baseline_features + [self.target_column, 'anonymous_patient_id']
        available_columns = [col for col in required_columns if col in df.columns]

        if len(available_columns) != len(required_columns):
            missing = set(required_columns) - set(available_columns)
            raise ValueError(f"Missing required columns for baseline features: {missing}")

        df_baseline = df[available_columns].copy()

        self.logger.info(f"Baseline feature set prepared with {len(self.baseline_features)} features")
        return df_baseline

    def prepare_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataset with enhanced features

        Args:
            df: Full featured dataframe

        Returns:
            pd.DataFrame: DataFrame with enhanced features + target
        """
        self.logger.info("Preparing enhanced feature set")

        # Required columns for enhanced model
        required_columns = self.enhanced_features + [self.target_column, 'anonymous_patient_id']

        # Check for missing temporal features and create defaults if needed
        if 'days_since_accident' not in df.columns:
            self.logger.warning("days_since_accident not found - creating default values")
            df = df.copy()
            df['days_since_accident'] = 14  # Default to 2 weeks

        if 'consultation_number' not in df.columns:
            self.logger.warning("consultation_number not found - creating default values")
            df = df.copy()
            df['consultation_number'] = 1  # Default to first consultation

        # Add interaction features to enhanced set
        enhanced_interaction_features = [
            'p_score_fl_score_interaction', 'severity_index',
            'both_improving', 'high_severity'
        ]

        # Select core enhanced features
        available_columns = [col for col in required_columns if col in df.columns]

        # Add available interaction features
        for feature in enhanced_interaction_features:
            if feature in df.columns:
                available_columns.append(feature)

        df_enhanced = df[available_columns].copy()

        self.logger.info(f"Enhanced feature set prepared with {len(available_columns) - 2} features")  # -2 for target and patient_id
        return df_enhanced

    def create_patient_level_splits(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[List[int], List[int]]:
        """
        Create patient-level train/test splits to prevent data leakage

        Args:
            df: DataFrame with anonymous_patient_id
            test_size: Proportion of patients for test set

        Returns:
            Tuple of train and test patient ID lists
        """
        self.logger.info(f"Creating patient-level train/test split ({(1-test_size)*100:.0f}%/{test_size*100:.0f}%)")

        # Get unique patients and their NBE distribution
        patient_nbe = df.groupby('anonymous_patient_id')[self.target_column].agg(lambda x: x.mode().iloc[0])

        # Stratify based on most common NBE value per patient
        train_patients, test_patients = train_test_split(
            patient_nbe.index.tolist(),
            test_size=test_size,
            stratify=patient_nbe.values,
            random_state=self.random_state
        )

        # Log split statistics
        train_nbe_dist = patient_nbe[train_patients].value_counts().to_dict()
        test_nbe_dist = patient_nbe[test_patients].value_counts().to_dict()

        split_stats = {
            'total_patients': len(patient_nbe),
            'train_patients': len(train_patients),
            'test_patients': len(test_patients),
            'train_nbe_distribution': train_nbe_dist,
            'test_nbe_distribution': test_nbe_dist
        }

        self.logger.info(f"Patient split: {len(train_patients)} train, {len(test_patients)} test")
        self.logger.info(f"Train NBE distribution: {train_nbe_dist}")
        self.logger.info(f"Test NBE distribution: {test_nbe_dist}")

        return train_patients, test_patients

    def split_dataset(self, df: pd.DataFrame, train_patients: List[int], test_patients: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset based on patient lists

        Args:
            df: DataFrame to split
            train_patients: List of patient IDs for training
            test_patients: List of patient IDs for testing

        Returns:
            Tuple of train and test dataframes
        """
        train_df = df[df['anonymous_patient_id'].isin(train_patients)].copy()
        test_df = df[df['anonymous_patient_id'].isin(test_patients)].copy()

        # Remove patient ID from feature sets (keep for reference)
        train_df_features = train_df.drop('anonymous_patient_id', axis=1)
        test_df_features = test_df.drop('anonymous_patient_id', axis=1)

        self.logger.info(f"Dataset split: {len(train_df)} train records, {len(test_df)} test records")

        return train_df_features, test_df_features

    def validate_feature_sets(self, baseline_train: pd.DataFrame, baseline_test: pd.DataFrame,
                             enhanced_train: pd.DataFrame, enhanced_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate prepared feature sets

        Args:
            baseline_train: Baseline training set
            baseline_test: Baseline test set
            enhanced_train: Enhanced training set
            enhanced_test: Enhanced test set

        Returns:
            Dict containing validation results
        """
        self.logger.info("Validating prepared feature sets")

        validation_results = {
            'is_valid': True,
            'baseline_validation': {},
            'enhanced_validation': {},
            'consistency_checks': {},
            'issues': []
        }

        # Baseline validation
        baseline_train_target_dist = baseline_train[self.target_column].value_counts().to_dict()
        baseline_test_target_dist = baseline_test[self.target_column].value_counts().to_dict()

        validation_results['baseline_validation'] = {
            'train_shape': baseline_train.shape,
            'test_shape': baseline_test.shape,
            'feature_count': len(self.baseline_features),
            'train_target_distribution': baseline_train_target_dist,
            'test_target_distribution': baseline_test_target_dist,
            'features_present': list(baseline_train.columns)
        }

        # Enhanced validation
        enhanced_train_target_dist = enhanced_train[self.target_column].value_counts().to_dict()
        enhanced_test_target_dist = enhanced_test[self.target_column].value_counts().to_dict()

        validation_results['enhanced_validation'] = {
            'train_shape': enhanced_train.shape,
            'test_shape': enhanced_test.shape,
            'feature_count': enhanced_train.shape[1] - 1,  # -1 for target
            'train_target_distribution': enhanced_train_target_dist,
            'test_target_distribution': enhanced_test_target_dist,
            'features_present': list(enhanced_train.columns)
        }

        # Consistency checks
        record_count_match = (baseline_train.shape[0] == enhanced_train.shape[0] and
                             baseline_test.shape[0] == enhanced_test.shape[0])

        target_dist_match = (baseline_train_target_dist == enhanced_train_target_dist and
                            baseline_test_target_dist == enhanced_test_target_dist)

        validation_results['consistency_checks'] = {
            'record_counts_match': record_count_match,
            'target_distributions_match': target_dist_match,
            'baseline_features_in_enhanced': all(feat in enhanced_train.columns for feat in self.baseline_features)
        }

        # Check for issues
        if not record_count_match:
            validation_results['issues'].append("Record counts don't match between baseline and enhanced sets")
            validation_results['is_valid'] = False

        if not target_dist_match:
            validation_results['issues'].append("Target distributions don't match between baseline and enhanced sets")
            validation_results['is_valid'] = False

        # Check for missing values
        for name, df_set in [('baseline_train', baseline_train), ('baseline_test', baseline_test),
                           ('enhanced_train', enhanced_train), ('enhanced_test', enhanced_test)]:
            missing_count = df_set.isnull().sum().sum()
            if missing_count > 0:
                validation_results['issues'].append(f"Missing values found in {name}: {missing_count}")
                validation_results['is_valid'] = False

        if validation_results['is_valid']:
            self.logger.info("All feature set validation checks passed")
        else:
            self.logger.error(f"Feature set validation failed: {validation_results['issues']}")

        return validation_results

    def save_processed_datasets(self, baseline_train: pd.DataFrame, baseline_test: pd.DataFrame,
                               enhanced_train: pd.DataFrame, enhanced_test: pd.DataFrame,
                               processing_stats: Dict[str, Any]) -> Dict[str, Path]:
        """
        Save all processed datasets and metadata

        Args:
            baseline_train: Baseline training set
            baseline_test: Baseline test set
            enhanced_train: Enhanced training set
            enhanced_test: Enhanced test set
            processing_stats: Processing statistics and metadata

        Returns:
            Dict containing paths to saved files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        processed_dir = self.data_path / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save baseline datasets
        baseline_train_file = processed_dir / f'step2_baseline_train_{timestamp}.csv'
        baseline_test_file = processed_dir / f'step2_baseline_test_{timestamp}.csv'
        baseline_train.to_csv(baseline_train_file, index=False)
        baseline_test.to_csv(baseline_test_file, index=False)
        saved_files['baseline_train'] = baseline_train_file
        saved_files['baseline_test'] = baseline_test_file

        # Save enhanced datasets
        enhanced_train_file = processed_dir / f'step2_enhanced_train_{timestamp}.csv'
        enhanced_test_file = processed_dir / f'step2_enhanced_test_{timestamp}.csv'
        enhanced_train.to_csv(enhanced_train_file, index=False)
        enhanced_test.to_csv(enhanced_test_file, index=False)
        saved_files['enhanced_train'] = enhanced_train_file
        saved_files['enhanced_test'] = enhanced_test_file

        # Save processing metadata
        metadata_file = processed_dir / f'step2_preprocessing_metadata_{timestamp}.json'
        with open(metadata_file, 'w') as f:
            json.dump(processing_stats, f, indent=2, default=str)
        saved_files['metadata'] = metadata_file

        self.logger.info(f"All processed datasets saved with timestamp: {timestamp}")

        return saved_files

    def process_dataset(self, input_filename: str = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """
        Main processing pipeline for dual feature engineering

        Args:
            input_filename: Optional specific input file

        Returns:
            Tuple containing datasets dict and processing statistics
        """
        self.logger.info("Starting dual feature preprocessing pipeline")

        try:
            # Load anonymized data
            df = self.load_anonymized_data(input_filename)

            # Create temporal features
            df = self.create_temporal_features(df)

            # Create consultation sequence features
            df = self.create_consultation_sequence_features(df)

            # Create interaction features
            df = self.create_interaction_features(df)

            # Create patient-level splits
            train_patients, test_patients = self.create_patient_level_splits(df)

            # Prepare baseline feature sets
            df_baseline = self.prepare_baseline_features(df)
            baseline_train, baseline_test = self.split_dataset(df_baseline, train_patients, test_patients)

            # Prepare enhanced feature sets
            df_enhanced = self.prepare_enhanced_features(df)
            enhanced_train, enhanced_test = self.split_dataset(df_enhanced, train_patients, test_patients)

            # Validate feature sets
            validation_results = self.validate_feature_sets(baseline_train, baseline_test, enhanced_train, enhanced_test)

            # Compile processing statistics
            processing_stats = {
                'timestamp': datetime.now().isoformat(),
                'pipeline_steps': [
                    'load_anonymized_data',
                    'create_temporal_features',
                    'create_consultation_sequence_features',
                    'create_interaction_features',
                    'patient_level_splits',
                    'prepare_feature_sets',
                    'validate_datasets'
                ],
                'dataset_info': {
                    'total_records': len(df),
                    'total_patients': df['anonymous_patient_id'].nunique(),
                    'train_patients': len(train_patients),
                    'test_patients': len(test_patients)
                },
                'feature_engineering': {
                    'baseline_features': self.baseline_features,
                    'enhanced_features': self.enhanced_features,
                    'interaction_features_created': [
                        'p_score_fl_score_interaction', 'severity_index',
                        'both_improving', 'both_worsening', 'high_severity', 'no_symptoms'
                    ],
                    'temporal_features_created': ['days_since_accident'],
                    'sequence_features_created': ['consultation_number', 'is_first_consultation', 'is_follow_up']
                },
                'data_splits': {
                    'train_test_ratio': '80/20',
                    'split_method': 'patient_level_stratified',
                    'random_state': self.random_state
                },
                'validation_results': validation_results
            }

            # Save datasets
            datasets = {
                'baseline_train': baseline_train,
                'baseline_test': baseline_test,
                'enhanced_train': enhanced_train,
                'enhanced_test': enhanced_test
            }

            saved_files = self.save_processed_datasets(
                baseline_train, baseline_test, enhanced_train, enhanced_test, processing_stats
            )

            processing_stats['saved_files'] = {str(k): str(v) for k, v in saved_files.items()}

            self.logger.info("Dual feature preprocessing pipeline completed successfully")
            self.logger.info(f"Datasets prepared for both baseline and enhanced models")

            return datasets, processing_stats

        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise