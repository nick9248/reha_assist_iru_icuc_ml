"""
Main script for Step 4: Model Training and Tuning
Orchestrates the hyperparameter optimization, cross-validation, and model refinement.
"""

import os
import logging
import argparse
from utils.project_setup import setup_logging, create_project_structure
from hyperparameter_tuning import run_hyperparameter_tuning
from cross_validation import run_patient_cross_validation
from model_refinement import run_model_refinement


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run model training and tuning (Step 4)")

    parser.add_argument("--datasets_path", type=str, help="Path to prepared datasets")
    parser.add_argument("--final_model_path", type=str, help="Path to final model from step 3")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for hyperparameter optimization")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout for hyperparameter optimization (seconds)")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of cross-validation folds")

    return parser.parse_args()


def find_latest_paths():
    """Find the latest datasets and model paths if not provided"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Try to get paths directly from environment variables
    direct_dataset_path = os.environ.get('PREPARED_DATASETS')
    direct_model_path = os.environ.get('FINAL_MODEL')

    if direct_dataset_path and os.path.exists(direct_dataset_path):
        print(f"Found datasets path from env: {direct_dataset_path}")
        return direct_dataset_path, direct_model_path

    # If not available directly, try to find in model directory
    model_base = os.environ.get('MODEL_FOLDER')
    if not model_base:
        print("MODEL_FOLDER environment variable is not set.")
        # Try common locations
        possible_locations = [
            'models',
            'C:/Users/Nick/PycharmProjects/reha_assist_iru/models',
            './models',
            '../models'
        ]

        for loc in possible_locations:
            if os.path.exists(loc):
                model_base = loc
                print(f"Found model directory at: {model_base}")
                break

    if not model_base or not os.path.exists(model_base):
        print(f"Could not find model directory at: {model_base}")
        return None, None

    # Try to find in model development directory
    model_dev_dir = os.path.join(model_base, "03_model_development")

    if not os.path.exists(model_dev_dir):
        print(f"Model development directory not found at: {model_dev_dir}")
        return None, None

    datasets_path = None
    final_model_path = None

    # Get all subdirectories
    try:
        subdirs = [os.path.join(model_dev_dir, d) for d in os.listdir(model_dev_dir)
                   if os.path.isdir(os.path.join(model_dev_dir, d))]

        if subdirs:
            # Find the latest subdirectory
            latest_subdir = max(subdirs, key=os.path.getmtime)
            print(f"Latest model directory: {latest_subdir}")

            # Check for datasets
            potential_datasets_path = os.path.join(latest_subdir, "prepared_datasets.pkl")
            if os.path.exists(potential_datasets_path):
                datasets_path = potential_datasets_path

            # Check for final model
            potential_model_path = os.path.join(latest_subdir, "final_model.pkl")
            if os.path.exists(potential_model_path):
                final_model_path = potential_model_path
    except Exception as e:
        print(f"Error finding paths: {str(e)}")

    # Print paths for debugging
    print(f"Found datasets path: {datasets_path}")
    print(f"Found model path: {final_model_path}")

    return datasets_path, final_model_path


def main():
    """Main function to run the complete pipeline"""
    # Load environment variables first
    from dotenv import load_dotenv
    load_dotenv()

    # Define paths directly (as a fallback)
    default_datasets_path = "C:\\Users\\Nick\\PycharmProjects\\reha_assist_iru\\models\\03_model_development\\20250515_135640\\prepared_datasets.pkl"
    default_model_path = "C:\\Users\\Nick\\PycharmProjects\\reha_assist_iru\\models\\03_model_development\\20250515_135640\\final_model.pkl"

    # Parse arguments
    args = parse_args()

    # Try to find paths if not provided
    if not args.datasets_path or not args.final_model_path:
        datasets_path, final_model_path = find_latest_paths()
        args.datasets_path = args.datasets_path or datasets_path or default_datasets_path
        args.final_model_path = args.final_model_path or final_model_path or default_model_path

    # Check if paths exist and provide informative error messages
    if not args.datasets_path:
        print("ERROR: Could not find datasets path. Please specify using --datasets_path")
        print("Available environment variables:")
        for key, value in os.environ.items():
            if 'FOLDER' in key or 'DIR' in key or 'PATH' in key:
                print(f"  {key}: {value}")
        raise ValueError("Datasets path not found")

    if not os.path.exists(args.datasets_path):
        raise ValueError(f"Datasets file does not exist at: {args.datasets_path}")

    if not args.final_model_path:
        raise ValueError("Final model path not found. Please specify using --final_model_path")

    if not os.path.exists(args.final_model_path):
        raise ValueError(f"Final model file does not exist at: {args.final_model_path}")

    print(f"Using datasets path: {args.datasets_path}")
    print(f"Using model path: {args.final_model_path}")

    # Create project structure
    log_dir, plot_dir, model_dir = create_project_structure()

    # Create output directories
    step4_dir = os.path.join(model_dir, "04_model_tuning")
    os.makedirs(step4_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(log_dir, logger_name="model_tuning")
    logger.info("=== MODEL TRAINING AND TUNING (STEP 4) ===")

    # Log arguments
    logger.info("Arguments:")
    logger.info(f"  datasets_path: {args.datasets_path}")
    logger.info(f"  final_model_path: {args.final_model_path}")
    logger.info(f"  n_trials: {args.n_trials}")
    logger.info(f"  timeout: {args.timeout}")
    logger.info(f"  cv_folds: {args.cv_folds}")

    # Run model refinement pipeline
    refinement_results = run_model_refinement(
        args.datasets_path,
        args.final_model_path,
        step4_dir,
        logger,
        n_trials=args.n_trials,
        timeout=args.timeout
    )

    # Log final results
    logger.info("=== MODEL TRAINING AND TUNING COMPLETE ===")
    logger.info(f"Best model: {refinement_results['best_model']}")
    logger.info(f"Results saved to: {step4_dir}")

    return refinement_results

if __name__ == "__main__":
    main()