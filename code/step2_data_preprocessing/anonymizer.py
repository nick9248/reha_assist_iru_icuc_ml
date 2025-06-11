"""
Anonymizer Module for NBE Prediction Project
Handles patient identifier anonymization with simple numbering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
from datetime import datetime
import json


class PatientAnonymizer:
    """
    Handles patient identifier anonymization using simple sequential numbering
    """

    def __init__(self, data_path: Path, log_path: Path):
        self.data_path = data_path
        self.log_path = log_path
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for anonymization operations"""
        logger = logging.getLogger('PatientAnonymizer')
        logger.setLevel(logging.INFO)

        # Create log directory
        log_dir = self.log_path / 'step2'
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'anonymizer_{timestamp}.log'

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(file_handler)

        return logger

    def create_patient_mapping(self, df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Create mapping between original accident numbers and anonymous sequential IDs

        Args:
            df: DataFrame containing accident_number column

        Returns:
            Tuple containing forward and reverse mapping dictionaries
        """
        self.logger.info("Creating patient ID mapping")

        if 'accident_number' not in df.columns:
            raise ValueError("accident_number column not found in dataset")

        # Get unique accident numbers
        unique_patients = df['accident_number'].unique()
        unique_patients.sort()  # Sort for consistent ordering

        # Create sequential mapping starting from 1
        forward_mapping = {}  # original_id -> anonymous_id
        reverse_mapping = {}  # anonymous_id -> original_id

        for idx, original_id in enumerate(unique_patients, start=1):
            forward_mapping[str(original_id)] = idx
            reverse_mapping[idx] = str(original_id)

        self.logger.info(f"Created mappings for {len(unique_patients)} unique patients")
        self.logger.info(f"Anonymous IDs range: 1 to {len(unique_patients)}")

        return forward_mapping, reverse_mapping

    def anonymize_patient_ids(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Replace accident_number with anonymous sequential IDs

        Args:
            df: DataFrame to anonymize

        Returns:
            Tuple containing anonymized dataframe and anonymization statistics
        """
        self.logger.info("Starting patient ID anonymization")

        # Create mapping
        forward_mapping, reverse_mapping = self.create_patient_mapping(df)

        # Create anonymized dataframe
        df_anon = df.copy()

        # Replace accident_number with anonymous IDs
        df_anon['anonymous_patient_id'] = df_anon['accident_number'].map(forward_mapping)

        # Verify all mappings were successful
        unmapped_count = df_anon['anonymous_patient_id'].isnull().sum()
        if unmapped_count > 0:
            self.logger.error(f"Failed to map {unmapped_count} patient IDs")
            raise ValueError(f"Anonymization failed for {unmapped_count} records")

        # Remove original accident_number column
        df_anon = df_anon.drop('accident_number', axis=1)

        # Reorder columns to put anonymous_patient_id first
        cols = ['anonymous_patient_id'] + [col for col in df_anon.columns if col != 'anonymous_patient_id']
        df_anon = df_anon[cols]

        # Generate anonymization statistics
        anonymization_stats = {
            'original_patients': len(forward_mapping),
            'anonymized_patients': df_anon['anonymous_patient_id'].nunique(),
            'total_records': len(df_anon),
            'anonymization_successful': unmapped_count == 0,
            'mapping_verification': {
                'forward_mapping_size': len(forward_mapping),
                'reverse_mapping_size': len(reverse_mapping),
                'id_range': f"1 to {max(reverse_mapping.keys())}",
                'mappings_consistent': len(forward_mapping) == len(reverse_mapping)
            }
        }

        self.logger.info(f"Anonymization completed successfully")
        self.logger.info(f"Anonymized {len(forward_mapping)} patients across {len(df_anon)} records")

        return df_anon, anonymization_stats, forward_mapping, reverse_mapping

    def validate_anonymization(self, df_original: pd.DataFrame, df_anonymized: pd.DataFrame,
                             forward_mapping: Dict[str, int]) -> Dict[str, Any]:
        """
        Validate that anonymization was performed correctly

        Args:
            df_original: Original dataframe with accident_number
            df_anonymized: Anonymized dataframe with anonymous_patient_id
            forward_mapping: Mapping from original to anonymous IDs

        Returns:
            Dict containing validation results
        """
        self.logger.info("Validating anonymization process")

        validation_results = {
            'is_valid': True,
            'checks': {},
            'issues': []
        }

        # Check 1: Record count consistency
        record_count_match = len(df_original) == len(df_anonymized)
        validation_results['checks']['record_count_match'] = record_count_match
        if not record_count_match:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Record count mismatch after anonymization")

        # Check 2: Patient count consistency
        original_patients = df_original['accident_number'].nunique()
        anonymized_patients = df_anonymized['anonymous_patient_id'].nunique()
        patient_count_match = original_patients == anonymized_patients
        validation_results['checks']['patient_count_match'] = patient_count_match
        if not patient_count_match:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Patient count mismatch after anonymization")

        # Check 3: No original IDs in anonymized data
        no_original_ids = 'accident_number' not in df_anonymized.columns
        validation_results['checks']['original_ids_removed'] = no_original_ids
        if not no_original_ids:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Original accident_number column still present")

        # Check 4: All anonymous IDs are valid integers
        valid_anonymous_ids = df_anonymized['anonymous_patient_id'].dtype in ['int64', 'int32']
        validation_results['checks']['valid_anonymous_ids'] = valid_anonymous_ids
        if not valid_anonymous_ids:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Anonymous IDs are not integers")

        # Check 5: Sequential numbering starts from 1
        min_id = df_anonymized['anonymous_patient_id'].min()
        max_id = df_anonymized['anonymous_patient_id'].max()
        expected_range = max_id == len(forward_mapping) and min_id == 1
        validation_results['checks']['sequential_numbering'] = expected_range
        if not expected_range:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Anonymous IDs are not sequential starting from 1")

        # Check 6: Mapping consistency
        mapping_consistent = len(set(forward_mapping.values())) == len(forward_mapping)
        validation_results['checks']['mapping_unique'] = mapping_consistent
        if not mapping_consistent:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Forward mapping contains duplicate values")

        # Check 7: All features preserved (except accident_number)
        expected_columns = set(df_original.columns) - {'accident_number'} | {'anonymous_patient_id'}
        actual_columns = set(df_anonymized.columns)
        columns_preserved = expected_columns == actual_columns
        validation_results['checks']['columns_preserved'] = columns_preserved
        if not columns_preserved:
            missing_cols = expected_columns - actual_columns
            extra_cols = actual_columns - expected_columns
            if missing_cols:
                validation_results['issues'].append(f"Missing columns: {missing_cols}")
            if extra_cols:
                validation_results['issues'].append(f"Unexpected columns: {extra_cols}")
            validation_results['is_valid'] = False

        # Summary
        validation_results['summary'] = {
            'total_checks': len(validation_results['checks']),
            'passed_checks': sum(validation_results['checks'].values()),
            'failed_checks': len(validation_results['checks']) - sum(validation_results['checks'].values()),
            'total_issues': len(validation_results['issues'])
        }

        if validation_results['is_valid']:
            self.logger.info("All anonymization validation checks passed")
        else:
            self.logger.error(f"Anonymization validation failed: {validation_results['issues']}")

        return validation_results

    def analyze_consultation_patterns(self, df_anonymized: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze consultation patterns in anonymized data

        Args:
            df_anonymized: Anonymized dataframe

        Returns:
            Dict containing consultation pattern analysis
        """
        self.logger.info("Analyzing consultation patterns")

        # Consultations per patient
        consultations_per_patient = df_anonymized.groupby('anonymous_patient_id').size()

        # Contact patterns over time
        contact_analysis = {}
        if 'contact_date' in df_anonymized.columns:
            df_temp = df_anonymized.copy()
            df_temp['contact_date'] = pd.to_datetime(df_temp['contact_date'])

            # Sort by patient and contact date
            df_temp = df_temp.sort_values(['anonymous_patient_id', 'contact_date'])

            # Calculate consultation sequence for each patient
            df_temp['consultation_sequence'] = df_temp.groupby('anonymous_patient_id').cumcount() + 1

            contact_analysis = {
                'sequence_distribution': df_temp['consultation_sequence'].value_counts().to_dict(),
                'max_consultations_per_patient': df_temp['consultation_sequence'].max(),
                'patients_with_multiple_consultations': (consultations_per_patient > 1).sum(),
                'single_consultation_patients': (consultations_per_patient == 1).sum()
            }

        patterns = {
            'consultation_distribution': consultations_per_patient.value_counts().to_dict(),
            'consultation_stats': {
                'mean_consultations_per_patient': consultations_per_patient.mean(),
                'median_consultations_per_patient': consultations_per_patient.median(),
                'min_consultations': consultations_per_patient.min(),
                'max_consultations': consultations_per_patient.max(),
                'std_consultations': consultations_per_patient.std()
            },
            'patient_categories': {
                'single_consultation': (consultations_per_patient == 1).sum(),
                'multiple_consultations': (consultations_per_patient > 1).sum(),
                'high_frequency_patients': (consultations_per_patient >= 5).sum()
            },
            'contact_patterns': contact_analysis
        }

        self.logger.info(f"Consultation analysis completed for {len(consultations_per_patient)} patients")

        return patterns

    def save_anonymization_artifacts(self, df_anonymized: pd.DataFrame,
                                   anonymization_stats: Dict[str, Any],
                                   forward_mapping: Dict[str, int],
                                   reverse_mapping: Dict[int, str],
                                   validation_results: Dict[str, Any],
                                   consultation_patterns: Dict[str, Any]) -> Tuple[Path, Path]:
        """
        Save anonymized data and all related artifacts

        Args:
            df_anonymized: Anonymized dataframe
            anonymization_stats: Anonymization statistics
            forward_mapping: Original to anonymous ID mapping
            reverse_mapping: Anonymous to original ID mapping
            validation_results: Validation results
            consultation_patterns: Consultation pattern analysis

        Returns:
            Tuple of paths to saved data file and mapping file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create directories
        processed_dir = self.data_path / 'processed'
        anonymized_dir = self.data_path / 'anonymized'
        processed_dir.mkdir(parents=True, exist_ok=True)
        anonymized_dir.mkdir(parents=True, exist_ok=True)

        # Save anonymized dataset
        data_file = processed_dir / f'step2_anonymized_data_{timestamp}.csv'
        df_anonymized.to_csv(data_file, index=False)

        # Save mapping table (secure storage)
        mapping_file = anonymized_dir / f'anonymization_mapping_{timestamp}.json'
        mapping_data = {
            'timestamp': timestamp,
            'forward_mapping': forward_mapping,
            'reverse_mapping': {str(k): v for k, v in reverse_mapping.items()},  # Convert keys to strings for JSON
            'anonymization_stats': anonymization_stats,
            'validation_results': validation_results,
            'consultation_patterns': consultation_patterns,
            'metadata': {
                'total_patients': len(forward_mapping),
                'total_records': len(df_anonymized),
                'anonymization_method': 'sequential_numbering',
                'id_range': f"1 to {len(forward_mapping)}"
            }
        }

        with open(mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2, default=str)

        self.logger.info(f"Anonymized data saved to: {data_file}")
        self.logger.info(f"Mapping table saved to: {mapping_file}")
        self.logger.warning("SECURITY NOTE: Mapping file contains sensitive information - store securely!")

        return data_file, mapping_file

    def anonymize_dataset(self, input_filename: str = None) -> Tuple[pd.DataFrame, Dict[str, Any], Path, Path]:
        """
        Main anonymization pipeline

        Args:
            input_filename: Optional specific input file from Step 2 cleaning

        Returns:
            Tuple containing anonymized dataframe, complete stats, data file path, and mapping file path
        """
        self.logger.info("Starting patient anonymization pipeline")

        try:
            # Load cleaned data from Step 2
            if input_filename:
                file_path = self.data_path / 'processed' / input_filename
            else:
                # Find most recent cleaned data file
                processed_dir = self.data_path / 'processed'
                cleaned_files = list(processed_dir.glob('step2_cleaned_data_*.csv'))
                if not cleaned_files:
                    raise FileNotFoundError("No Step 2 cleaned data files found")
                file_path = max(cleaned_files, key=lambda x: x.stat().st_mtime)

            self.logger.info(f"Loading cleaned data from: {file_path}")
            df = pd.read_csv(file_path)

            # Perform anonymization
            df_anonymized, anonymization_stats, forward_mapping, reverse_mapping = self.anonymize_patient_ids(df)

            # Validate anonymization
            validation_results = self.validate_anonymization(df, df_anonymized, forward_mapping)

            # Analyze consultation patterns
            consultation_patterns = self.analyze_consultation_patterns(df_anonymized)

            # Save all artifacts
            data_file, mapping_file = self.save_anonymization_artifacts(
                df_anonymized, anonymization_stats, forward_mapping,
                reverse_mapping, validation_results, consultation_patterns
            )

            # Compile complete statistics
            complete_stats = {
                'anonymization': anonymization_stats,
                'validation': validation_results,
                'consultation_patterns': consultation_patterns,
                'files': {
                    'anonymized_data': str(data_file),
                    'mapping_table': str(mapping_file)
                }
            }

            self.logger.info("Patient anonymization pipeline completed successfully")

            return df_anonymized, complete_stats, data_file, mapping_file

        except Exception as e:
            self.logger.error(f"Error in anonymization pipeline: {str(e)}")
            raise