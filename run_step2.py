"""
Step 2 Main Orchestrator for NBE Prediction Project
Coordinates data cleaning, anonymization, and preprocessing pipeline
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import Step 2 modules (will be created)
try:
    from code.step2_data_preprocessing.data_cleaner import DataCleaner
    from code.step2_data_preprocessing.anonymizer import PatientAnonymizer
    from code.step2_data_preprocessing.preprocessor import DualFeaturePreprocessor
except ImportError as e:
    print(f"⚠️  Step 2 modules not found: {e}")
    print("📝 Please create the step2_data_preprocessing module first")
    sys.exit(1)

# Optional config imports (create basic versions if missing)
try:
    from config.logging_config import setup_logging
except ImportError:
    def setup_logging():
        pass

try:
    from utils.datetime_utils import get_timestamp
except ImportError:
    def get_timestamp():
        return datetime.now().strftime('%Y%m%d_%H%M%S')


class Step2Orchestrator:
    """
    Orchestrates the complete Step 2 pipeline: cleaning, anonymization, and preprocessing
    """

    def __init__(self, project_root: Path = None):
        # Auto-detect project root if not provided
        if project_root is None:
            # Get the directory where this script is located (should be project root)
            project_root = Path(__file__).parent.absolute()

        self.project_root = project_root
        self.data_path = project_root / 'data'
        self.log_path = project_root / 'logs'

        # Debug: Print paths for verification
        print(f"🔍 Project root: {self.project_root}")
        print(f"🔍 Data path: {self.data_path}")
        print(f"🔍 Data path exists: {self.data_path.exists()}")

        if self.data_path.exists():
            processed_path = self.data_path / 'processed'
            print(f"🔍 Processed path: {processed_path}")
            print(f"🔍 Processed path exists: {processed_path.exists()}")

            if processed_path.exists():
                files = list(processed_path.glob('*'))
                print(f"🔍 Files in processed: {files}")

        # Setup logging
        self.logger = self._setup_logger()

        # Initialize components
        self.data_cleaner = DataCleaner(self.data_path, self.log_path)
        self.anonymizer = PatientAnonymizer(self.data_path, self.log_path)
        self.preprocessor = DualFeaturePreprocessor(self.data_path, self.log_path)

    def _setup_logger(self) -> logging.Logger:
        """Setup main orchestrator logger"""
        logger = logging.getLogger('Step2Orchestrator')
        logger.setLevel(logging.INFO)

        # Create log directory
        log_dir = self.log_path / 'step2'
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'step2_orchestrator_{timestamp}.log'

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger

    def run_data_cleaning(self) -> tuple:
        """
        Execute data cleaning pipeline

        Returns:
            Tuple of cleaned dataframe and cleaning summary
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 2.1: DATA CLEANING")
        self.logger.info("=" * 60)

        try:
            cleaned_df, cleaning_summary = self.data_cleaner.clean_dataset()

            # Save cleaned data
            cleaned_file = self.data_cleaner.save_cleaned_data(cleaned_df, cleaning_summary)

            self.logger.info(f"✅ Data cleaning completed successfully")
            self.logger.info(f"📊 Final dataset: {len(cleaned_df)} records from {cleaned_df['accident_number'].nunique()} patients")
            self.logger.info(f"💾 Saved to: {cleaned_file}")

            return cleaned_df, cleaning_summary, cleaned_file

        except Exception as e:
            self.logger.error(f"❌ Data cleaning failed: {str(e)}")
            raise

    def run_anonymization(self, cleaned_file_path: Path = None) -> tuple:
        """
        Execute patient anonymization pipeline

        Args:
            cleaned_file_path: Path to cleaned data file

        Returns:
            Tuple of anonymized dataframe, stats, data file, and mapping file
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 2.2: PATIENT ANONYMIZATION")
        self.logger.info("=" * 60)

        try:
            input_filename = cleaned_file_path.name if cleaned_file_path else None
            anonymized_df, anon_stats, data_file, mapping_file = self.anonymizer.anonymize_dataset(input_filename)

            self.logger.info(f"✅ Patient anonymization completed successfully")
            self.logger.info(f"Anonymized {anon_stats['anonymization']['original_patients']} patients")
            self.logger.info(f"Patient IDs: 1 to {anon_stats['anonymization']['original_patients']}")
            self.logger.info(f"Data saved to: {data_file}")
            self.logger.info(f"Mapping saved to: {mapping_file}")

            return anonymized_df, anon_stats, data_file, mapping_file

        except Exception as e:
            self.logger.error(f"❌ Patient anonymization failed: {str(e)}")
            raise

    def run_preprocessing(self, anonymized_file_path: Path = None) -> tuple:
        """
        Execute dual feature preprocessing pipeline

        Args:
            anonymized_file_path: Path to anonymized data file

        Returns:
            Tuple of datasets dictionary and processing stats
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 2.3: DUAL FEATURE PREPROCESSING")
        self.logger.info("=" * 60)

        try:
            input_filename = anonymized_file_path.name if anonymized_file_path else None
            datasets, processing_stats = self.preprocessor.process_dataset(input_filename)

            self.logger.info(f"✅ Dual feature preprocessing completed successfully")
            self.logger.info(f"Baseline model: {len(self.preprocessor.baseline_features)} features")
            self.logger.info(f"Enhanced model: {len(self.preprocessor.enhanced_features)} + interactions")
            self.logger.info(f"Train/Test split: 80%/20% at patient level")

            # Log dataset shapes
            for name, df in datasets.items():
                self.logger.info(f"{name}: {df.shape}")

            return datasets, processing_stats

        except Exception as e:
            self.logger.error(f"❌ Dual feature preprocessing failed: {str(e)}")
            raise

    def generate_step2_summary(self, cleaning_summary: dict, anon_stats: dict,
                              processing_stats: dict) -> dict:
        """
        Generate comprehensive Step 2 summary report

        Args:
            cleaning_summary: Data cleaning summary
            anon_stats: Anonymization statistics
            processing_stats: Preprocessing statistics

        Returns:
            Dict containing comprehensive Step 2 summary
        """
        self.logger.info("Generating Step 2 comprehensive summary")

        step2_summary = {
            'step_info': {
                'step_number': 2,
                'step_name': 'Data Preprocessing & Anonymization',
                'completion_timestamp': datetime.now().isoformat(),
                'status': 'COMPLETED_SUCCESSFULLY'
            },
            'pipeline_overview': {
                'components_executed': [
                    'data_cleaning',
                    'patient_anonymization',
                    'dual_feature_preprocessing'
                ],
                'total_execution_time': 'calculated_at_runtime',
                'data_flow': {
                    'input_source': 'Step 1 data exploration results',
                    'intermediate_outputs': [
                        'cleaned_data',
                        'anonymized_data',
                        'baseline_feature_sets',
                        'enhanced_feature_sets'
                    ],
                    'final_outputs': [
                        'baseline_train_test_splits',
                        'enhanced_train_test_splits'
                    ]
                }
            },
            'data_transformation': {
                'cleaning_phase': {
                    'original_records': cleaning_summary['data_flow']['original_records'],
                    'final_records': cleaning_summary['data_flow']['final_records'],
                    'records_removed': cleaning_summary['data_flow']['total_records_removed'],
                    'retention_rate': cleaning_summary['data_flow']['data_retention_rate'],
                    'nbe_binary_conversion': {
                        'removed_no_info_cases': cleaning_summary['nbe_filtering']['removed_records'],
                        'final_class_distribution': cleaning_summary['nbe_filtering']['final_distribution']
                    }
                },
                'anonymization_phase': {
                    'patients_anonymized': anon_stats['anonymization']['original_patients'],
                    'anonymization_method': 'sequential_numbering_1_to_n',
                    'consultation_relationships_preserved': True,
                    'validation_passed': anon_stats['validation']['is_valid']
                },
                'preprocessing_phase': {
                    'feature_engineering_completed': True,
                    'baseline_features': processing_stats['feature_engineering']['baseline_features'],
                    'enhanced_features': processing_stats['feature_engineering']['enhanced_features'],
                    'interaction_features': processing_stats['feature_engineering']['interaction_features_created'],
                    'temporal_features': processing_stats['feature_engineering']['temporal_features_created'],
                    'patient_level_splitting': True
                }
            },
            'model_readiness': {
                'baseline_model': {
                    'feature_count': len(processing_stats['feature_engineering']['baseline_features']),
                    'train_records': processing_stats['validation_results']['baseline_validation']['train_shape'][0],
                    'test_records': processing_stats['validation_results']['baseline_validation']['test_shape'][0],
                    'target_distribution_train': processing_stats['validation_results']['baseline_validation']['train_target_distribution'],
                    'target_distribution_test': processing_stats['validation_results']['baseline_validation']['test_target_distribution']
                },
                'enhanced_model': {
                    'feature_count': processing_stats['validation_results']['enhanced_validation']['feature_count'],
                    'train_records': processing_stats['validation_results']['enhanced_validation']['train_shape'][0],
                    'test_records': processing_stats['validation_results']['enhanced_validation']['test_shape'][0],
                    'target_distribution_train': processing_stats['validation_results']['enhanced_validation']['train_target_distribution'],
                    'target_distribution_test': processing_stats['validation_results']['enhanced_validation']['test_target_distribution']
                }
            },
            'quality_assurance': {
                'data_cleaning_passed': cleaning_summary['final_data_quality']['missing_values'] == 0,
                'anonymization_validation_passed': anon_stats['validation']['is_valid'],
                'feature_set_validation_passed': processing_stats['validation_results']['is_valid'],
                'patient_level_split_verified': processing_stats['validation_results']['consistency_checks']['record_counts_match'],
                'target_distribution_preserved': processing_stats['validation_results']['consistency_checks']['target_distributions_match']
            },
            'api_preparation': {
                'baseline_api_ready': {
                    'required_features': processing_stats['feature_engineering']['baseline_features'],
                    'feature_types': 'all_integer_0_to_4_or_0_to_2',
                    'preprocessing_required': 'none'
                },
                'enhanced_api_ready': {
                    'required_features': processing_stats['feature_engineering']['enhanced_features'],
                    'optional_features': ['days_since_accident', 'consultation_number'],
                    'default_values': {
                        'days_since_accident': 14,
                        'consultation_number': 1
                    },
                    'preprocessing_required': 'feature_interactions_calculated_at_runtime'
                }
            },
            'files_generated': processing_stats['saved_files'],
            'next_steps': {
                'step_3_ready': True,
                'recommended_actions': [
                    'Proceed to Step 3: Feature Engineering (additional interactions)',
                    'Begin Step 4: Model Training with both baseline and enhanced datasets',
                    'Validate temporal feature importance',
                    'Consider additional domain-specific features'
                ]
            }
        }

        return step2_summary

    def save_step2_summary(self, summary: dict) -> Path:
        """
        Save Step 2 comprehensive summary

        Args:
            summary: Step 2 summary dictionary

        Returns:
            Path to saved summary file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = self.data_path / 'processed' / f'step2_complete_summary_{timestamp}.json'

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Step 2 summary saved to: {summary_file}")
        return summary_file

    def run_complete_step2_pipeline(self) -> dict:
        """
        Execute the complete Step 2 pipeline

        Returns:
            Dict containing all results and file paths
        """
        start_time = datetime.now()

        self.logger.info("STARTING STEP 2: DATA PREPROCESSING & ANONYMIZATION PIPELINE")
        self.logger.info("=" * 80)

        try:
            # Step 2.1: Data Cleaning
            cleaned_df, cleaning_summary, cleaned_file = self.run_data_cleaning()

            # Step 2.2: Patient Anonymization
            anonymized_df, anon_stats, anon_data_file, mapping_file = self.run_anonymization(cleaned_file)

            # Step 2.3: Dual Feature Preprocessing
            datasets, processing_stats = self.run_preprocessing(anon_data_file)

            # Generate comprehensive summary
            step2_summary = self.generate_step2_summary(cleaning_summary, anon_stats, processing_stats)

            # Calculate execution time
            end_time = datetime.now()
            execution_time = end_time - start_time
            step2_summary['pipeline_overview']['total_execution_time'] = str(execution_time)

            # Save summary
            summary_file = self.save_step2_summary(step2_summary)

            # Final success message
            self.logger.info("=" * 80)
            self.logger.info("STEP 2 PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            self.logger.info(f"Total execution time: {execution_time}")
            self.logger.info(f"Final datasets ready for model training:")
            self.logger.info(f"   • Baseline model: {datasets['baseline_train'].shape[0]} train + {datasets['baseline_test'].shape[0]} test")
            self.logger.info(f"   • Enhanced model: {datasets['enhanced_train'].shape[0]} train + {datasets['enhanced_test'].shape[0]} test")
            self.logger.info(f"All outputs saved to: {self.data_path / 'processed'}")
            self.logger.info("Ready for Step 3: Advanced Feature Engineering")

            return {
                'status': 'SUCCESS',
                'execution_time': str(execution_time),
                'datasets': datasets,
                'cleaning_summary': cleaning_summary,
                'anonymization_stats': anon_stats,
                'processing_stats': processing_stats,
                'step2_summary': step2_summary,
                'files': {
                    'cleaned_data': str(cleaned_file),
                    'anonymized_data': str(anon_data_file),
                    'mapping_table': str(mapping_file),
                    'summary': str(summary_file)
                }
            }

        except Exception as e:
            self.logger.error("STEP 2 PIPELINE FAILED!")
            self.logger.error(f"Error: {str(e)}")
            raise

    def print_step2_overview(self):
        """Print Step 2 pipeline overview"""
        overview = """
╔══════════════════════════════════════════════════════════════╗
║                    STEP 2 PIPELINE OVERVIEW                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📋 STEP 2.1: DATA CLEANING                                 ║
║     • Load Step 1 results                                   ║
║     • Remove NBE=2 cases (binary classification)            ║
║     • Validate data quality                                 ║
║     • Handle missing values & duplicates                    ║
║                                                              ║
║  🔐 STEP 2.2: PATIENT ANONYMIZATION                        ║
║     • Replace accident_number with sequential IDs (1,2,3..) ║
║     • Create translation table                              ║
║     • Preserve consultation relationships                   ║
║     • Validate anonymization integrity                      ║
║                                                              ║
║  ⚙️  STEP 2.3: DUAL FEATURE PREPROCESSING                   ║
║     • Create temporal features (days_since_accident)        ║
║     • Generate consultation sequences                       ║
║     • Build interaction features                            ║
║     • Prepare baseline (4-feature) datasets                 ║
║     • Prepare enhanced (6+ feature) datasets                ║
║     • Patient-level train/test splits (80/20)               ║
║                                                              ║
║  🎯 OUTPUTS:                                                ║
║     • Baseline train/test sets (API v1)                     ║
║     • Enhanced train/test sets (API v2)                     ║
║     • Patient anonymization mapping                         ║
║     • Comprehensive preprocessing metadata                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(overview)


def main():
    """Main execution function"""

    # Auto-detect project root (directory where this script is located)
    project_root = Path(__file__).parent.absolute()
    print(f"Auto-detected project root: {project_root}")

    # Initialize orchestrator
    orchestrator = Step2Orchestrator(project_root)

    # Print overview
    orchestrator.print_step2_overview()

    try:
        # Run complete pipeline directly
        results = orchestrator.run_complete_step2_pipeline()

        print("\nStep 2 completed successfully!")
        print(f"Check {project_root / 'data' / 'processed'} for all output files")
        print("Ready to proceed to Step 3: Advanced Feature Engineering")

        return results

    except Exception as e:
        print(f"\nStep 2 failed: {str(e)}")
        return None


if __name__ == "__main__":
    results = main()