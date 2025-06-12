"""
API Validator Module for NBE Prediction Project
Handles business rule validation and input checking
"""

from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class APIValidator:
    """
    Validator for API requests and business rules
    """

    def __init__(self):
        # Business rules from Step 2 data preprocessing
        self.feature_ranges = {
            'p_score': {'min': 0, 'max': 4, 'description': 'Pain score'},
            'p_status': {'min': 0, 'max': 2, 'description': 'Pain status'},
            'fl_score': {'min': 0, 'max': 4, 'description': 'Function limitation score'},
            'fl_status': {'min': 0, 'max': 2, 'description': 'Function limitation status'},
            'days_since_accident': {'min': 0, 'max': 1000, 'description': 'Days since accident'},
            'consultation_number': {'min': 1, 'max': 20, 'description': 'Consultation sequence number'}
        }

        # Valid status combinations based on medical logic
        self.valid_status_combinations = {
            # (p_status, fl_status): description
            (0, 0): "Both pain and function worsening",
            (0, 1): "Pain worsening, function stable",
            (0, 2): "Pain worsening, function improving",
            (1, 0): "Pain stable, function worsening",
            (1, 1): "Both pain and function stable",
            (1, 2): "Pain stable, function improving",
            (2, 0): "Pain improving, function worsening",
            (2, 1): "Pain improving, function stable",
            (2, 2): "Both pain and function improving"
        }

    def validate_feature_ranges(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that all features are within expected ranges

        Args:
            data: Request data dictionary

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        for feature, value in data.items():
            if feature in self.feature_ranges:
                rules = self.feature_ranges[feature]

                if not isinstance(value, (int, float)):
                    errors.append(f"{feature} must be a numeric value")
                    continue

                if value < rules['min'] or value > rules['max']:
                    errors.append(
                        f"{feature} ({rules['description']}) must be between "
                        f"{rules['min']} and {rules['max']}, got {value}"
                    )

        return len(errors) == 0, errors

    def validate_business_logic(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate business logic and medical rules

        Args:
            data: Request data dictionary

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Validate pain and function scores relationship
        p_score = data.get('p_score', 0)
        fl_score = data.get('fl_score', 0)

        # Logical consistency: very high pain usually correlates with high function limitation
        if p_score == 4 and fl_score == 0:
            errors.append(
                "Warning: Maximum pain (4) with no function limitation (0) is medically unusual"
            )

        # Status values logic
        p_status = data.get('p_status')
        fl_status = data.get('fl_status')

        if p_status is not None and fl_status is not None:
            status_combo = (p_status, fl_status)
            if status_combo not in self.valid_status_combinations:
                errors.append(f"Invalid status combination: p_status={p_status}, fl_status={fl_status}")

        # Temporal logic validation
        if 'days_since_accident' in data and 'consultation_number' in data:
            days = data['days_since_accident']
            consultation_num = data['consultation_number']

            # First consultation shouldn't be too long after accident
            if consultation_num == 1 and days > 90:
                errors.append(
                    f"Warning: First consultation {days} days after accident is unusually late"
                )

            # Multiple consultations should have reasonable spacing
            if consultation_num > 1 and days < 7:
                errors.append(
                    f"Warning: Consultation #{consultation_num} only {days} days after accident seems early"
                )

        return len(errors) == 0, errors

    def validate_temporal_consistency(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate temporal feature consistency

        Args:
            data: Request data dictionary

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        days_since_accident = data.get('days_since_accident')
        consultation_number = data.get('consultation_number')

        if days_since_accident is not None and consultation_number is not None:
            # Estimate reasonable time per consultation
            avg_days_per_consultation = days_since_accident / consultation_number

            # Flag unrealistic scenarios
            if avg_days_per_consultation < 1:
                errors.append(
                    f"Temporal inconsistency: {consultation_number} consultations in {days_since_accident} days "
                    f"(avg {avg_days_per_consultation:.1f} days per consultation) is too frequent"
                )
            elif avg_days_per_consultation > 90:
                errors.append(
                    f"Temporal inconsistency: {consultation_number} consultations in {days_since_accident} days "
                    f"(avg {avg_days_per_consultation:.1f} days per consultation) has large gaps"
                )

        return len(errors) == 0, errors

    def validate_request(self, data: Dict[str, Any], endpoint_type: str = 'baseline') -> Dict[str, Any]:
        """
        Comprehensive request validation

        Args:
            data: Request data dictionary
            endpoint_type: 'baseline' or 'enhanced'

        Returns:
            Dict containing validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'validation_timestamp': datetime.now().isoformat()
        }

        try:
            # 1. Feature range validation
            range_valid, range_errors = self.validate_feature_ranges(data)
            if not range_valid:
                validation_result['errors'].extend(range_errors)
                validation_result['is_valid'] = False

            # 2. Business logic validation
            logic_valid, logic_errors = self.validate_business_logic(data)
            if not logic_valid:
                # Treat business logic issues as warnings unless critical
                critical_errors = [e for e in logic_errors if not e.startswith('Warning:')]
                warnings = [e for e in logic_errors if e.startswith('Warning:')]

                validation_result['errors'].extend(critical_errors)
                validation_result['warnings'].extend(warnings)

                if critical_errors:
                    validation_result['is_valid'] = False

            # 3. Enhanced endpoint specific validation
            if endpoint_type == 'enhanced':
                temporal_valid, temporal_errors = self.validate_temporal_consistency(data)
                if not temporal_valid:
                    # Temporal issues are usually warnings
                    validation_result['warnings'].extend(temporal_errors)

            # 4. Log validation results
            if validation_result['errors']:
                logger.warning(f"Validation errors for {endpoint_type} endpoint: {validation_result['errors']}")

            if validation_result['warnings']:
                logger.info(f"Validation warnings for {endpoint_type} endpoint: {validation_result['warnings']}")

            if validation_result['is_valid']:
                logger.debug(f"Validation passed for {endpoint_type} endpoint")

        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation system error: {str(e)}")

        return validation_result

    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about valid feature ranges and rules

        Returns:
            Dict containing feature information
        """
        return {
            'feature_ranges': self.feature_ranges,
            'valid_status_combinations': {
                f"p_status={combo[0]}, fl_status={combo[1]}": desc
                for combo, desc in self.valid_status_combinations.items()
            },
            'business_rules': [
                "Pain score (p_score) should generally correlate with function limitation (fl_score)",
                "Status values indicate change vs previous consultation",
                "First consultation typically occurs within 90 days of accident",
                "Multiple consultations should have reasonable spacing (7+ days apart)",
                "Very high pain (4) with no function limitation (0) is medically unusual"
            ]
        }