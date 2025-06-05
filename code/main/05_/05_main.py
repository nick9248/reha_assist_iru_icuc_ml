"""
Main script for Step 5: Model Interpretation and Insights
Orchestrates the feature importance analysis and probability calibration.
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
from utils.project_setup import setup_logging, create_project_structure
from feature_importance_analysis import analyze_feature_importance, load_data, load_model
from probability_calibration import run_probability_calibration_analysis


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run model interpretation and insights (Step 5)")

    parser.add_argument("--datasets_path", type=str, help="Path to prepared datasets")
    parser.add_argument("--model_path", type=str, help="Path to refined model")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples for SHAP calculation")
    parser.add_argument("--no_calibrate", action="store_true", help="Skip model calibration")

    return parser.parse_args()


def find_latest_paths():
    """Find the latest datasets and model paths if not provided"""
    model_base = os.environ.get('MODEL_FOLDER', 'models')

    # Try to find model in Step 4 (model tuning) directory first
    model_tuning_dir = os.path.join(model_base, "04_model_tuning")
    datasets_path = None
    model_path = None

    if os.path.exists(model_tuning_dir):
        subdirs = [os.path.join(model_tuning_dir, d) for d in os.listdir(model_tuning_dir)
                   if os.path.isdir(os.path.join(model_tuning_dir, d))]
        if subdirs:
            latest_subdir = max(subdirs, key=os.path.getmtime)

            # Try to find the refined model
            for model_name in ['refined_model.pkl', 'refined_model_prod.joblib', 'final_model.pkl']:
                potential_model_path = os.path.join(latest_subdir, '04_model_tuning', model_name)
                if os.path.exists(potential_model_path):
                    model_path = potential_model_path
                    break

    # If not found in Step 4, try Step 3
    if not model_path:
        model_dev_dir = os.path.join(model_base, "03_model_development")
        if os.path.exists(model_dev_dir):
            subdirs = [os.path.join(model_dev_dir, d) for d in os.listdir(model_dev_dir)
                       if os.path.isdir(os.path.join(model_dev_dir, d))]
            if subdirs:
                latest_subdir = max(subdirs, key=os.path.getmtime)

                # Try to find the final model
                model_path = os.path.join(latest_subdir, "final_model.pkl")
                if not os.path.exists(model_path):
                    model_path = os.path.join(latest_subdir, "final_model_prod.joblib")

    # Look for datasets in Step 3
    model_dev_dir = os.path.join(model_base, "03_model_development")
    if os.path.exists(model_dev_dir):
        subdirs = [os.path.join(model_dev_dir, d) for d in os.listdir(model_dev_dir)
                   if os.path.isdir(os.path.join(model_dev_dir, d))]
        if subdirs:
            latest_subdir = max(subdirs, key=os.path.getmtime)
            datasets_path = os.path.join(latest_subdir, "prepared_datasets.pkl")

    return datasets_path, model_path


def extract_feature_importance_summary(importance_df, top_n=5):
    """
    Extract a clean summary of feature importance

    Parameters:
    -----------
    importance_df : pandas.DataFrame
        Feature importance dataframe
    top_n : int
        Number of top features to include

    Returns:
    --------
    list
        List of tuples (feature_name, importance, direction)
    """
    if importance_df is None or len(importance_df) == 0:
        return []

    # Get top features
    top_df = importance_df.head(top_n)

    # Create summary list
    summary = []
    for _, row in top_df.iterrows():
        feature = row['Feature']
        importance = row['Importance']
        direction = row['Direction'] if 'Direction' in row else 'Unknown'

        summary.append((feature, importance, direction))

    return summary


def main():
    """Main function to run the complete pipeline"""
    # Parse arguments
    args = parse_args()

    # Find latest paths if not provided
    if not args.datasets_path or not args.model_path:
        datasets_path, model_path = find_latest_paths()
        args.datasets_path = args.datasets_path or datasets_path
        args.model_path = args.model_path or model_path

    # Check if paths exist
    if not args.datasets_path or not os.path.exists(args.datasets_path):
        raise ValueError(f"Datasets not found at {args.datasets_path}")
    if not args.model_path or not os.path.exists(args.model_path):
        raise ValueError(f"Model not found at {args.model_path}")

    # Create project structure
    log_dir, plot_dir, model_dir = create_project_structure()

    # Create output directories
    step5_dir = os.path.join(model_dir, "05_model_interpretation")
    os.makedirs(step5_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(log_dir, logger_name="model_interpretation")
    logger.info("=== MODEL INTERPRETATION AND INSIGHTS (STEP 5) ===")

    # Log arguments
    logger.info("Arguments:")
    logger.info(f"  datasets_path: {args.datasets_path}")
    logger.info(f"  model_path: {args.model_path}")
    logger.info(f"  n_samples: {args.n_samples}")
    logger.info(f"  no_calibrate: {args.no_calibrate}")

    # Load datasets and model
    datasets = load_data(args.datasets_path, logger)
    model = load_model(args.model_path, logger)

    # Step 1: Feature Importance Analysis
    logger.info("Step 1: Feature Importance Analysis")
    feature_dir = os.path.join(step5_dir, "feature_importance")
    os.makedirs(feature_dir, exist_ok=True)

    importance_df = analyze_feature_importance(model, datasets, feature_dir, logger, args.n_samples)

    # Step 2: Probability Calibration
    logger.info("Step 2: Probability Calibration Analysis")
    calibration_dir = os.path.join(step5_dir, "probability_calibration")
    os.makedirs(calibration_dir, exist_ok=True)

    calibration_results = run_probability_calibration_analysis(
        model, datasets, calibration_dir, logger, not args.no_calibrate
    )

    # Log completion
    logger.info("=== MODEL INTERPRETATION AND INSIGHTS COMPLETE ===")
    logger.info(f"Feature importance results saved to: {feature_dir}")
    logger.info(f"Probability calibration results saved to: {calibration_dir}")

    # Extract feature importance summary
    feature_summary = extract_feature_importance_summary(importance_df)

    # Create summary dictionary
    summary = {
        'feature_importance': {
            'top_features': [f[0] for f in feature_summary],
            'top_importance_values': [f[1] for f in feature_summary],
            'directions': [f[2] for f in feature_summary]
        },
        'probability_calibration': {}
    }

    # Add probability calibration stats if available
    if calibration_results and 'original_model' in calibration_results:
        if 'probability_stats' in calibration_results['original_model']:
            summary['probability_calibration']['original_brier_score'] = calibration_results['original_model']['probability_stats']['overall']['mean']

        # Add calibration improvement if performed
        if not args.no_calibrate and 'calibration' in calibration_results:
            best_method = calibration_results['calibration']['best_method']

            if best_method in calibration_results['calibration']:
                cal_results = calibration_results['calibration'][best_method]

                if 'calibrated' in cal_results and 'brier_score' in cal_results['calibrated']:
                    summary['probability_calibration']['calibrated_brier_score'] = cal_results['calibrated']['brier_score']

                if 'improvement' in cal_results and 'brier_score_pct' in cal_results['improvement']:
                    summary['probability_calibration']['improvement_percent'] = cal_results['improvement']['brier_score_pct']

                summary['probability_calibration']['calibration_method'] = best_method

    # Print summary of findings
    logger.info("\nSummary of findings:")

    # Feature importance summary
    logger.info("Top 5 most important features:")
    for i, (feature, importance, direction) in enumerate(feature_summary):
        logger.info(f"  {i + 1}. {feature} ({importance:.4f}, {direction})")

    # Probability calibration summary
    logger.info("\nProbability calibration:")
    if 'original_brier_score' in summary['probability_calibration']:
        logger.info(f"  Original Brier score: {summary['probability_calibration']['original_brier_score']:.4f}")

    if 'calibrated_brier_score' in summary['probability_calibration']:
        logger.info(f"  Calibrated Brier score: {summary['probability_calibration']['calibrated_brier_score']:.4f}")
        logger.info(f"  Improvement: {summary['probability_calibration']['improvement_percent']:.2f}%")
        logger.info(f"  Calibration method: {summary['probability_calibration']['calibration_method']}")

    # Save summary to file
    import json
    with open(os.path.join(step5_dir, 'summary.json'), 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        json_summary = {}
        for key, value in summary.items():
            if isinstance(value, dict):
                json_summary[key] = {}
                for k, v in value.items():
                    if isinstance(v, (list, tuple)):
                        json_summary[key][k] = [float(x) if isinstance(x, np.number) else x for x in v]
                    elif isinstance(v, np.number):
                        json_summary[key][k] = float(v)
                    else:
                        json_summary[key][k] = v
            else:
                json_summary[key] = value

        json.dump(json_summary, f, indent=4)

    logger.info(f"Summary saved to: {os.path.join(step5_dir, 'summary.json')}")

    return summary


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    # Run main function
    main()