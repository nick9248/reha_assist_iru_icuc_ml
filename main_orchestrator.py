"""
Step 1: Data Exploration & Understanding - Main Orchestrator
Coordinates data loading, exploration, and validation for NBE prediction project
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
import logging
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from code.step1_data_exploration.data_loader import DataLoader
from code.step1_data_exploration.data_explorer import DataExplorer
from code.step1_data_exploration.data_validator import DataValidator


class Step1Orchestrator:
    """
    Orchestrates Step 1: Data Exploration & Understanding
    """

    def __init__(self):
        # Setup paths from environment or use defaults
        self.project_root = Path(__file__).parent.parent
        self.data_path = self.project_root / 'data'
        self.logs_path = self.project_root / 'logs'
        self.plots_path = self.project_root / 'plots'

        # Create directories if they don't exist
        self._create_directories()

        # Setup main logger
        self.logger = self._setup_logger()

        # Initialize components
        self.data_loader = DataLoader(self.data_path, self.logs_path)
        self.data_explorer = DataExplorer(self.plots_path, self.logs_path)
        self.data_validator = DataValidator(self.logs_path)

    def _create_directories(self):
        """Create necessary project directories"""
        directories = [
            self.data_path / 'raw',
            self.data_path / 'processed',
            self.logs_path / 'step1',
            self.plots_path / 'step1_data_exploration'
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Setup main orchestrator logger"""
        logger = logging.getLogger('Step1Orchestrator')
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.logs_path / 'step1' / f'step1_orchestrator_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers
        if not logger.handlers:
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger

    def run_data_exploration(self, filename: str = 'icuc_ml_dataset.xlsx') -> Dict[str, Any]:
        """
        Execute complete data exploration pipeline

        Args:
            filename: Name of the dataset file

        Returns:
            Dict containing all exploration results
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING STEP 1: DATA EXPLORATION & UNDERSTANDING")
        self.logger.info("=" * 60)

        exploration_results = {
            'timestamp': datetime.now().isoformat(),
            'step': 'step1_data_exploration',
            'status': 'started'
        }

        try:
            # Phase 1: Data Loading
            self.logger.info("Phase 1: Loading dataset")
            df = self.data_loader.load_excel_data(filename)
            exploration_results['data_loading'] = {
                'status': 'success',
                'shape': df.shape,
                'columns': list(df.columns)
            }
            self.logger.info(f"✓ Data loaded successfully: {df.shape}")

            # Phase 2: Initial Data Quality Assessment
            self.logger.info("Phase 2: Initial data quality assessment")
            schema_validation = self.data_loader.validate_data_schema(df)
            quality_metrics = self.data_loader.log_data_quality_metrics(df)

            exploration_results['initial_assessment'] = {
                'schema_validation': schema_validation,
                'quality_metrics': quality_metrics
            }
            self.logger.info(f"✓ Initial assessment completed")

            # Phase 3: Comprehensive Data Validation
            self.logger.info("Phase 3: Comprehensive data validation")
            validation_report = self.data_validator.run_comprehensive_validation(df)
            exploration_results['validation_report'] = validation_report

            # Log validation summary
            overall_status = validation_report['overall_status']
            self.logger.info(f"✓ Validation completed - Valid: {overall_status['is_valid']}")
            if overall_status['critical_issues']:
                self.logger.warning(f"Critical issues found: {overall_status['critical_issues']}")

            # Phase 4: Exploratory Data Analysis & Visualization
            self.logger.info("Phase 4: Generating exploratory visualizations")
            plot_paths = self.data_explorer.generate_comprehensive_report(df)
            exploration_results['visualizations'] = plot_paths
            self.logger.info(f"✓ Visualizations generated: {len([p for p in plot_paths.values() if p])}")

            # Phase 5: Save Results
            self.logger.info("Phase 5: Saving exploration results")
            output_file = self.data_loader.save_exploration_results(
                df, schema_validation, quality_metrics
            )
            exploration_results['output_file'] = str(output_file)
            self.logger.info(f"✓ Results saved to: {output_file}")

            # Phase 6: Generate Summary Report
            self.logger.info("Phase 6: Generating summary report")
            summary_report = self._generate_summary_report(exploration_results)
            exploration_results['summary_report'] = summary_report

            # Final status
            exploration_results['status'] = 'completed'
            self.logger.info("=" * 60)
            self.logger.info("STEP 1 COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)

            # Print summary to console
            self._print_summary_to_console(summary_report)

        except Exception as e:
            exploration_results['status'] = 'failed'
            exploration_results['error'] = str(e)
            self.logger.error(f"Step 1 failed: {str(e)}", exc_info=True)
            raise

        return exploration_results

    def _generate_summary_report(self, exploration_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive summary report

        Args:
            exploration_results: Complete exploration results

        Returns:
            Dict containing summary report
        """
        summary = {
            'dataset_overview': {},
            'data_quality_summary': {},
            'validation_summary': {},
            'ml_readiness_summary': {},
            'key_findings': [],
            'recommendations': [],
            'next_steps': []
        }

        # Dataset Overview
        if 'data_loading' in exploration_results:
            summary['dataset_overview'] = {
                'total_rows': exploration_results['data_loading']['shape'][0],
                'total_columns': exploration_results['data_loading']['shape'][1],
                'columns': exploration_results['data_loading']['columns']
            }

        # Data Quality Summary
        if 'initial_assessment' in exploration_results:
            quality_metrics = exploration_results['initial_assessment']['quality_metrics']
            summary['data_quality_summary'] = {
                'missing_values_total': sum([
                    v['count'] for v in quality_metrics.get('missing_values', {}).values()
                ]),
                'duplicate_rows': quality_metrics.get('basic_stats', {}).get('duplicate_rows', 0),
                'unique_patients': quality_metrics.get('patient_analysis', {}).get('unique_patients', 'N/A'),
                'memory_usage_mb': quality_metrics.get('basic_stats', {}).get('memory_usage_mb', 0)
            }

        # Validation Summary
        if 'validation_report' in exploration_results:
            validation = exploration_results['validation_report']
            summary['validation_summary'] = {
                'schema_compliant': validation['schema_compliance']['is_compliant'],
                'business_rules_valid': validation['business_rules']['is_valid'],
                'data_integrity_intact': validation['data_integrity']['is_intact'],
                'ml_ready': validation['ml_readiness']['is_ready'],
                'overall_valid': validation['overall_status']['is_valid'],
                'critical_issues_count': len(validation['overall_status']['critical_issues'])
            }

        # ML Readiness Summary
        if 'validation_report' in exploration_results:
            ml_readiness = exploration_results['validation_report']['ml_readiness']
            summary['ml_readiness_summary'] = {
                'readiness_score': ml_readiness['summary']['readiness_score'],
                'sample_size_adequate': ml_readiness['sample_size_check']['is_adequate'],
                'target_suitable': ml_readiness.get('target_quality', {}).get('is_suitable', True),
                'class_balance_ok': not ml_readiness.get('class_balance', {}).get('needs_balancing', False)
            }

        # Key Findings
        summary['key_findings'] = self._extract_key_findings(exploration_results)

        # Recommendations
        summary['recommendations'] = self._extract_recommendations(exploration_results)

        # Next Steps
        summary['next_steps'] = self._define_next_steps(exploration_results)

        return summary

    def _extract_key_findings(self, exploration_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from exploration results"""
        findings = []

        # Data shape finding
        if 'data_loading' in exploration_results:
            shape = exploration_results['data_loading']['shape']
            findings.append(f"Dataset contains {shape[0]:,} records with {shape[1]} features")

        # Patient analysis finding
        if 'initial_assessment' in exploration_results:
            patient_analysis = exploration_results['initial_assessment']['quality_metrics'].get('patient_analysis', {})
            if 'unique_patients' in patient_analysis:
                unique_patients = patient_analysis['unique_patients']
                consultations_per_patient = patient_analysis.get('consultations_per_patient', {})
                avg_consultations = consultations_per_patient.get('mean', 0)
                findings.append(
                    f"Data covers {unique_patients:,} unique patients with average {avg_consultations:.1f} consultations per patient")

        # Target variable finding
        if 'initial_assessment' in exploration_results:
            target_analysis = exploration_results['initial_assessment']['quality_metrics'].get('target_analysis', {})
            if 'class_distribution' in target_analysis:
                class_dist = target_analysis['class_distribution']
                findings.append(f"Target variable (NBE) distribution: {class_dist}")

        # Data quality finding
        if 'validation_report' in exploration_results:
            validation = exploration_results['validation_report']
            if validation['overall_status']['is_valid']:
                findings.append("Data passes all validation checks and is ready for preprocessing")
            else:
                issues = len(validation['overall_status']['critical_issues'])
                findings.append(f"Data has {issues} critical issues that need attention")

        return findings

    def _extract_recommendations(self, exploration_results: Dict[str, Any]) -> List[str]:
        """Extract recommendations from exploration results"""
        recommendations = []

        # From validation report
        if 'validation_report' in exploration_results:
            validation_recs = exploration_results['validation_report']['overall_status'].get('recommendations', [])
            recommendations.extend(validation_recs)

        # Additional recommendations based on findings
        if 'initial_assessment' in exploration_results:
            quality_metrics = exploration_results['initial_assessment']['quality_metrics']

            # Missing values recommendation
            missing_values = quality_metrics.get('missing_values', {})
            if missing_values:
                recommendations.append("Address missing values before model training")

            # Duplicate records recommendation
            duplicates = quality_metrics.get('basic_stats', {}).get('duplicate_rows', 0)
            if duplicates > 0:
                recommendations.append(f"Investigate and handle {duplicates} duplicate records")

        return recommendations

    def _define_next_steps(self, exploration_results: Dict[str, Any]) -> List[str]:
        """Define next steps based on exploration results"""
        next_steps = []

        # Check if data is ready for next step
        if 'validation_report' in exploration_results:
            validation = exploration_results['validation_report']

            if validation['overall_status']['is_valid']:
                next_steps.extend([
                    "Proceed to Step 2: Data Preprocessing & Anonymization",
                    "Implement data cleaning strategies if needed",
                    "Create anonymized patient identifiers",
                    "Prepare data for feature engineering"
                ])
            else:
                next_steps.extend([
                    "Address critical data issues identified in validation",
                    "Re-run data exploration after fixing issues",
                    "Consult with domain experts for business rule clarification"
                ])

        # Always include documentation step
        next_steps.append("Document findings and share with stakeholders")

        return next_steps

    def _print_summary_to_console(self, summary_report: Dict[str, Any]):
        """Print formatted summary to console"""
        print("\n" + "=" * 80)
        print("STEP 1: DATA EXPLORATION & UNDERSTANDING - SUMMARY REPORT")
        print("=" * 80)

        # Dataset Overview
        print("\n📊 DATASET OVERVIEW:")
        overview = summary_report['dataset_overview']
        print(f"   • Total Records: {overview.get('total_rows', 'N/A'):,}")
        print(f"   • Total Features: {overview.get('total_columns', 'N/A')}")
        print(f"   • Columns: {', '.join(overview.get('columns', []))}")

        # Data Quality Summary
        print("\n🔍 DATA QUALITY SUMMARY:")
        quality = summary_report['data_quality_summary']
        print(f"   • Missing Values: {quality.get('missing_values_total', 'N/A')}")
        print(f"   • Duplicate Records: {quality.get('duplicate_rows', 'N/A')}")
        print(f"   • Unique Patients: {quality.get('unique_patients', 'N/A')}")
        print(f"   • Memory Usage: {quality.get('memory_usage_mb', 'N/A'):.2f} MB")

        # Validation Summary
        print("\n✅ VALIDATION SUMMARY:")
        validation = summary_report['validation_summary']
        print(f"   • Schema Compliant: {'✓' if validation.get('schema_compliant') else '✗'}")
        print(f"   • Business Rules Valid: {'✓' if validation.get('business_rules_valid') else '✗'}")
        print(f"   • Data Integrity: {'✓' if validation.get('data_integrity_intact') else '✗'}")
        print(f"   • ML Ready: {'✓' if validation.get('ml_ready') else '✗'}")
        print(f"   • Overall Status: {'✓ PASSED' if validation.get('overall_valid') else '✗ FAILED'}")

        # ML Readiness
        print("\n🤖 ML READINESS SUMMARY:")
        ml_readiness = summary_report['ml_readiness_summary']
        print(f"   • Readiness Score: {ml_readiness.get('readiness_score', 0):.1f}/100")
        print(f"   • Sample Size: {'✓' if ml_readiness.get('sample_size_adequate') else '✗'}")
        print(f"   • Target Quality: {'✓' if ml_readiness.get('target_suitable') else '✗'}")
        print(f"   • Class Balance: {'✓' if ml_readiness.get('class_balance_ok') else '✗'}")

        # Key Findings
        print("\n🔑 KEY FINDINGS:")
        for i, finding in enumerate(summary_report['key_findings'], 1):
            print(f"   {i}. {finding}")

        # Recommendations
        if summary_report['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for i, recommendation in enumerate(summary_report['recommendations'], 1):
                print(f"   {i}. {recommendation}")

        # Next Steps
        print("\n➡️  NEXT STEPS:")
        for i, step in enumerate(summary_report['next_steps'], 1):
            print(f"   {i}. {step}")

        print("\n" + "=" * 80)
        print("END OF SUMMARY REPORT")
        print("=" * 80 + "\n")


def main():
    """
    Main execution function for Step 1
    """
    try:
        # Initialize orchestrator
        orchestrator = Step1Orchestrator()

        # Check if data file exists
        data_file = orchestrator.data_path / 'raw' / 'icuc_ml_dataset.xlsx'
        if not data_file.exists():
            print(f"❌ Error: Data file not found at {data_file}")
            print("Please place the 'icuc_ml_dataset.xlsx' file in the data/raw/ directory")
            return

        # Run data exploration
        results = orchestrator.run_data_exploration()

        # Save final results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = orchestrator.project_root / f'step1_results_{timestamp}.json'

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📁 Complete results saved to: {results_file}")

        return results

    except Exception as e:
        print(f"❌ Step 1 execution failed: {str(e)}")
        logging.error(f"Step 1 execution failed: {str(e)}", exc_info=True)
        return None


if __name__ == "__main__":
    results = main()