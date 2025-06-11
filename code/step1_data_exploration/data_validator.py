"""
Data Validator Module for NBE Prediction Project
Comprehensive data validation and quality checks
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import json

class DataValidator:
    """
    Comprehensive data validation for NBE dataset
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.logger = self._setup_logger()

        # Business rules for NBE prediction
        self.business_rules = {
            'feature_ranges': {
                'p_score': {'min': 0, 'max': 4, 'description': 'Pain score'},
                'p_status': {'min': 0, 'max': 2, 'description': 'Pain status'},
                'fl_score': {'min': 0, 'max': 4, 'description': 'Function limitation score'},
                'fl_status': {'min': 0, 'max': 2, 'description': 'Function limitation status'},
                'nbe': {'min': 0, 'max': 2, 'description': 'NBE target variable'}
            },
            'required_columns': [
                'accident_number', 'p_score', 'p_status', 'fl_score', 'fl_status', 'nbe'
            ],
            'data_types': {
                'accident_number': ['object', 'string'],
                'p_score': ['int64', 'int32', 'float64'],
                'p_status': ['int64', 'int32', 'float64'],
                'fl_score': ['int64', 'int32', 'float64'],
                'fl_status': ['int64', 'int32', 'float64'],
                'nbe': ['int64', 'int32', 'float64']
            }
        }

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for data validation operations"""
        logger = logging.getLogger('DataValidator')
        logger.setLevel(logging.INFO)

        # Create log directory
        log_dir = self.log_path / 'step1'
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'data_validator_{timestamp}.log'

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(file_handler)

        return logger

    def validate_schema_compliance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate dataset schema against business requirements

        Args:
            df: DataFrame to validate

        Returns:
            Dict containing schema validation results
        """
        self.logger.info("Starting schema compliance validation")

        validation_results = {
            'is_compliant': True,
            'missing_columns': [],
            'extra_columns': [],
            'type_issues': {},
            'summary': {}
        }

        # Check required columns
        missing_cols = set(self.business_rules['required_columns']) - set(df.columns)
        if missing_cols:
            validation_results['missing_columns'] = list(missing_cols)
            validation_results['is_compliant'] = False
            self.logger.error(f"Missing required columns: {missing_cols}")

        # Check for extra columns (informational)
        extra_cols = set(df.columns) - set(self.business_rules['required_columns'])
        if extra_cols:
            validation_results['extra_columns'] = list(extra_cols)
            self.logger.info(f"Extra columns found: {extra_cols}")

        # Validate data types
        for col in self.business_rules['required_columns']:
            if col in df.columns:
                actual_type = str(df[col].dtype)
                expected_types = self.business_rules['data_types'][col]

                if actual_type not in expected_types:
                    validation_results['type_issues'][col] = {
                        'actual': actual_type,
                        'expected': expected_types,
                        'can_convert': self._check_type_convertibility(df[col], expected_types)
                    }
                    self.logger.warning(f"Type mismatch in {col}: {actual_type} not in {expected_types}")

        validation_results['summary'] = {
            'total_columns': len(df.columns),
            'required_columns_present': len(self.business_rules['required_columns']) - len(missing_cols),
            'type_issues_count': len(validation_results['type_issues'])
        }

        self.logger.info(f"Schema validation completed. Compliant: {validation_results['is_compliant']}")
        return validation_results

    def validate_business_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate business rules and constraints

        Args:
            df: DataFrame to validate

        Returns:
            Dict containing business rule validation results
        """
        self.logger.info("Starting business rules validation")

        validation_results = {
            'is_valid': True,
            'range_violations': {},
            'logical_inconsistencies': {},
            'data_quality_issues': {},
            'summary': {}
        }

        # Validate feature ranges
        for feature, rules in self.business_rules['feature_ranges'].items():
            if feature in df.columns:
                violations = self._check_range_violations(df[feature], rules['min'], rules['max'])
                if violations['count'] > 0:
                    validation_results['range_violations'][feature] = violations
                    validation_results['is_valid'] = False
                    self.logger.warning(f"Range violations in {feature}: {violations['count']} cases")

        # Check logical inconsistencies
        logical_issues = self._check_logical_consistency(df)
        if logical_issues:
            validation_results['logical_inconsistencies'] = logical_issues
            validation_results['is_valid'] = False
            self.logger.warning(f"Logical inconsistencies found: {len(logical_issues)} types")

        # Data quality checks
        quality_issues = self._check_data_quality(df)
        validation_results['data_quality_issues'] = quality_issues

        validation_results['summary'] = {
            'total_rows': len(df),
            'total_violations': sum([v['count'] for v in validation_results['range_violations'].values()]),
            'violation_rate': self._calculate_violation_rate(df, validation_results['range_violations'])
        }

        self.logger.info(f"Business rules validation completed. Valid: {validation_results['is_valid']}")
        return validation_results

    def validate_data_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data integrity and consistency

        Args:
            df: DataFrame to validate

        Returns:
            Dict containing data integrity validation results
        """
        self.logger.info("Starting data integrity validation")

        integrity_results = {
            'is_intact': True,
            'duplicate_records': {},
            'patient_consistency': {},
            'temporal_consistency': {},
            'summary': {}
        }

        # Check for duplicate records
        duplicate_info = self._check_duplicates(df)
        integrity_results['duplicate_records'] = duplicate_info
        if duplicate_info['total_duplicates'] > 0:
            integrity_results['is_intact'] = False
            self.logger.warning(f"Found {duplicate_info['total_duplicates']} duplicate records")

        # Patient-level consistency checks
        if 'accident_number' in df.columns:
            patient_consistency = self._check_patient_consistency(df)
            integrity_results['patient_consistency'] = patient_consistency
            if not patient_consistency['is_consistent']:
                integrity_results['is_intact'] = False
                self.logger.warning("Patient-level inconsistencies detected")

        integrity_results['summary'] = {
            'total_records': len(df),
            'unique_records': len(df) - duplicate_info['total_duplicates'],
            'integrity_score': self._calculate_integrity_score(integrity_results)
        }

        self.logger.info(f"Data integrity validation completed. Intact: {integrity_results['is_intact']}")
        return integrity_results

    def validate_ml_readiness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate if data is ready for machine learning

        Args:
            df: DataFrame to validate

        Returns:
            Dict containing ML readiness validation results
        """
        self.logger.info("Starting ML readiness validation")

        ml_readiness = {
            'is_ready': True,
            'sample_size_check': {},
            'feature_quality': {},
            'target_quality': {},
            'class_balance': {},
            'recommendations': [],
            'summary': {}
        }

        # Sample size adequacy
        sample_check = self._check_sample_size_adequacy(df)
        ml_readiness['sample_size_check'] = sample_check
        if not sample_check['is_adequate']:
            ml_readiness['is_ready'] = False
            ml_readiness['recommendations'].append("Increase sample size for better model performance")

        # Feature quality assessment
        feature_quality = self._assess_feature_quality(df)
        ml_readiness['feature_quality'] = feature_quality

        # Target variable quality
        if 'nbe' in df.columns:
            target_quality = self._assess_target_quality(df['nbe'])
            ml_readiness['target_quality'] = target_quality

            if not target_quality['is_suitable']:
                ml_readiness['is_ready'] = False
                ml_readiness['recommendations'].extend(target_quality['recommendations'])

        # Class balance analysis
        if 'nbe' in df.columns:
            class_balance = self._analyze_class_balance(df['nbe'])
            ml_readiness['class_balance'] = class_balance

            if class_balance['needs_balancing']:
                ml_readiness['recommendations'].append("Consider class balancing techniques")

        ml_readiness['summary'] = {
            'readiness_score': self._calculate_readiness_score(ml_readiness),
            'critical_issues': len([r for r in ml_readiness['recommendations'] if 'critical' in r.lower()]),
            'total_recommendations': len(ml_readiness['recommendations'])
        }

        self.logger.info(f"ML readiness validation completed. Ready: {ml_readiness['is_ready']}")
        return ml_readiness

    def _check_type_convertibility(self, series: pd.Series, expected_types: List[str]) -> bool:
        """Check if series can be converted to expected types"""
        try:
            if any(t in ['int64', 'int32'] for t in expected_types):
                pd.to_numeric(series, errors='raise')
                return True
        except:
            return False
        return False

    def _check_range_violations(self, series: pd.Series, min_val: int, max_val: int) -> Dict:
        """Check for values outside valid range"""
        # Handle missing values explicitly for pandas compatibility
        valid_mask = series.between(min_val, max_val) | series.isnull()
        violations = series[~valid_mask]
        return {
            'count': len(violations),
            'percentage': len(violations) / len(series) * 100,
            'invalid_values': violations.unique().tolist() if len(violations) > 0 else []
        }

    def _check_logical_consistency(self, df: pd.DataFrame) -> Dict:
        """Check for logical inconsistencies in the data"""
        inconsistencies = {}

        # Example: Check if status values make sense for score values
        # This is domain-specific and can be expanded based on business logic

        return inconsistencies

    def _check_data_quality(self, df: pd.DataFrame) -> Dict:
        """Comprehensive data quality assessment"""
        quality_issues = {
            'missing_values': {},
            'outliers': {},
            'suspicious_patterns': {}
        }

        # Missing values analysis
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                quality_issues['missing_values'][col] = {
                    'count': missing_count,
                    'percentage': missing_count / len(df) * 100
                }

        # Outlier detection for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outliers = self._detect_outliers(df[col])
            if len(outliers) > 0:
                quality_issues['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100,
                    'values': outliers.tolist()
                }

        return quality_issues

    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series[(series < lower_bound) | (series > upper_bound)]

    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate records"""
        total_duplicates = df.duplicated().sum()
        duplicate_subset = df.duplicated(subset=['accident_number'], keep=False).sum() if 'accident_number' in df.columns else 0

        return {
            'total_duplicates': total_duplicates,
            'patient_level_duplicates': duplicate_subset,
            'duplicate_percentage': total_duplicates / len(df) * 100
        }

    def _check_patient_consistency(self, df: pd.DataFrame) -> Dict:
        """Check consistency within patient records"""
        # This would check for inconsistencies in patient data across consultations
        # Implementation depends on specific business rules
        return {'is_consistent': True, 'issues': []}

    def _check_sample_size_adequacy(self, df: pd.DataFrame) -> Dict:
        """Check if sample size is adequate for ML"""
        min_samples_per_class = 30  # Rule of thumb
        total_samples = len(df)

        adequacy = {
            'total_samples': total_samples,
            'is_adequate': total_samples >= 100,  # Minimum for basic ML
            'recommendation': 'adequate' if total_samples >= 500 else 'marginal' if total_samples >= 100 else 'insufficient'
        }

        return adequacy

    def _assess_feature_quality(self, df: pd.DataFrame) -> Dict:
        """Assess quality of features for ML"""
        features = ['p_score', 'p_status', 'fl_score', 'fl_status']
        available_features = [f for f in features if f in df.columns]

        quality_assessment = {
            'total_features': len(available_features),
            'feature_completeness': {},
            'feature_variance': {}
        }

        for feature in available_features:
            # Completeness
            completeness = (1 - df[feature].isnull().sum() / len(df)) * 100
            quality_assessment['feature_completeness'][feature] = completeness

            # Variance check
            variance = df[feature].var()
            quality_assessment['feature_variance'][feature] = variance

        return quality_assessment

    def _assess_target_quality(self, target_series: pd.Series) -> Dict:
        """Assess target variable quality"""
        quality = {
            'is_suitable': True,
            'completeness': (1 - target_series.isnull().sum() / len(target_series)) * 100,
            'class_count': target_series.nunique(),
            'recommendations': []
        }

        if quality['completeness'] < 95:
            quality['is_suitable'] = False
            quality['recommendations'].append("Target variable has too many missing values")

        if quality['class_count'] < 2:
            quality['is_suitable'] = False
            quality['recommendations'].append("Target variable needs at least 2 classes")

        return quality

    def _analyze_class_balance(self, target_series: pd.Series) -> Dict:
        """Analyze class balance in target variable"""
        class_counts = target_series.value_counts()
        class_proportions = class_counts / len(target_series)

        min_proportion = class_proportions.min()
        balance_threshold = 0.1  # 10% minimum for minority class

        return {
            'class_distribution': class_counts.to_dict(),
            'class_proportions': class_proportions.to_dict(),
            'min_class_proportion': min_proportion,
            'needs_balancing': min_proportion < balance_threshold,
            'balance_ratio': class_proportions.max() / class_proportions.min()
        }

    def _calculate_violation_rate(self, df: pd.DataFrame, range_violations: Dict) -> float:
        """Calculate overall violation rate"""
        total_violations = sum([v['count'] for v in range_violations.values()])
        total_cells = len(df) * len(range_violations)
        return (total_violations / total_cells * 100) if total_cells > 0 else 0

    def _calculate_integrity_score(self, integrity_results: Dict) -> float:
        """Calculate overall data integrity score (0-100)"""
        base_score = 100

        # Deduct points for duplicates
        duplicate_penalty = integrity_results['duplicate_records']['duplicate_percentage']

        return max(0, base_score - duplicate_penalty)

    def _calculate_readiness_score(self, ml_readiness: Dict) -> float:
        """Calculate ML readiness score (0-100)"""
        score = 100

        # Sample size component
        if not ml_readiness['sample_size_check']['is_adequate']:
            score -= 30

        # Target quality component
        if 'target_quality' in ml_readiness and not ml_readiness['target_quality']['is_suitable']:
            score -= 40

        # Feature quality component
        feature_quality = ml_readiness.get('feature_quality', {})
        avg_completeness = np.mean(list(feature_quality.get('feature_completeness', {}).values()) or [100])
        score -= (100 - avg_completeness) * 0.3

        return max(0, score)

    def run_comprehensive_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all validation checks and generate comprehensive report

        Args:
            df: DataFrame to validate

        Returns:
            Dict containing all validation results
        """
        self.logger.info("Starting comprehensive data validation")

        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict()
            }
        }

        try:
            # Run all validation components
            validation_report['schema_compliance'] = self.validate_schema_compliance(df)
            validation_report['business_rules'] = self.validate_business_rules(df)
            validation_report['data_integrity'] = self.validate_data_integrity(df)
            validation_report['ml_readiness'] = self.validate_ml_readiness(df)

            # Overall assessment
            validation_report['overall_status'] = {
                'is_valid': all([
                    validation_report['schema_compliance']['is_compliant'],
                    validation_report['business_rules']['is_valid'],
                    validation_report['data_integrity']['is_intact'],
                    validation_report['ml_readiness']['is_ready']
                ]),
                'critical_issues': self._identify_critical_issues(validation_report),
                'recommendations': self._generate_overall_recommendations(validation_report)
            }

            self.logger.info("Comprehensive validation completed successfully")

        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            raise

        return validation_report

    def _identify_critical_issues(self, validation_report: Dict) -> List[str]:
        """Identify critical issues that must be addressed"""
        critical_issues = []

        if not validation_report['schema_compliance']['is_compliant']:
            critical_issues.append("Schema compliance failures")

        if not validation_report['business_rules']['is_valid']:
            critical_issues.append("Business rule violations")

        if not validation_report['ml_readiness']['is_ready']:
            critical_issues.append("Not ready for machine learning")

        return critical_issues

    def _generate_overall_recommendations(self, validation_report: Dict) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []

        # Add ML readiness recommendations
        if 'ml_readiness' in validation_report:
            recommendations.extend(validation_report['ml_readiness'].get('recommendations', []))

        # Add data quality recommendations
        if validation_report['business_rules']['range_violations']:
            recommendations.append("Clean data range violations before modeling")

        if validation_report['data_integrity']['duplicate_records']['total_duplicates'] > 0:
            recommendations.append("Remove or investigate duplicate records")

        return recommendations