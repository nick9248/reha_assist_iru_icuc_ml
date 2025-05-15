"""
Main script for the model development pipeline.
Orchestrates the complete pipeline including:
1. Data preparation
2. Baseline models
3. Advanced models
4. Model evaluation
5. Model calibration
6. Model selection and final testing
"""

import os
import logging
from utils.project_setup import create_project_structure, setup_logging
from utils.data_loader import load_engineered_dataset

# Import pipeline stages
from data_preparation import prepare_datasets
from baseline_models import train_baseline_models
from advanced_models import train_advanced_models
from model_evaluation import evaluate_models
from model_calibration import calibrate_best_model
from model_selection import select_best_model, create_final_test_plots


def run_pipeline():
    """Run the complete model development pipeline"""

    # Setup project structure and logging
    log_dir, plot_dir, model_dir = create_project_structure()
    logger = setup_logging(log_dir, "model_development_pipeline")

    logger.info("=== STARTING MODEL DEVELOPMENT PIPELINE ===")

    # Step 1: Data Preparation
    logger.info("Step 1: Data Preparation")
    df = load_engineered_dataset(logger)
    datasets = prepare_datasets(df, logger, apply_scaling=True, handle_imbalance=True)

    # Save prepared datasets
    import pickle
    datasets_path = os.path.join(model_dir, "prepared_datasets.pkl")
    with open(datasets_path, "wb") as f:
        pickle.dump(datasets, f)
    logger.info(f"Prepared datasets saved to: {datasets_path}")

    # Step 2: Train Baseline Models
    logger.info("Step 2: Training Baseline Models")
    baseline_models = train_baseline_models(datasets, logger)

    # Save baseline models
    baseline_models_path = os.path.join(model_dir, "baseline_models.pkl")
    with open(baseline_models_path, "wb") as f:
        pickle.dump(baseline_models, f)
    logger.info(f"Baseline models saved to: {baseline_models_path}")

    # Step 3: Train Advanced Models
    logger.info("Step 3: Training Advanced Models")
    advanced_models = train_advanced_models(datasets, logger)

    # Save advanced models
    advanced_models_path = os.path.join(model_dir, "advanced_models.pkl")
    with open(advanced_models_path, "wb") as f:
        pickle.dump(advanced_models, f)
    logger.info(f"Advanced models saved to: {advanced_models_path}")

    # Step 4: Model Evaluation
    logger.info("Step 4: Model Evaluation")
    all_models = evaluate_models(baseline_models, advanced_models, datasets, plot_dir, logger)

    # Save all models
    all_models_path = os.path.join(model_dir, "all_models.pkl")
    with open(all_models_path, "wb") as f:
        pickle.dump(all_models, f)
    logger.info(f"All models saved to: {all_models_path}")

    # Step 5: Model Calibration
    logger.info("Step 5: Model Calibration")
    best_model_name, calibrated_model, calibration_metrics = calibrate_best_model(all_models, datasets, logger)

    # Save calibrated model
    calibrated_model_info = {
        'model_name': best_model_name,
        'calibrated_model': calibrated_model,
        'calibration_metrics': calibration_metrics
    }

    calibrated_model_path = os.path.join(model_dir, "calibrated_model.pkl")
    with open(calibrated_model_path, "wb") as f:
        pickle.dump(calibrated_model_info, f)
    logger.info(f"Calibrated model saved to: {calibrated_model_path}")

    # Step 6: Final Model Selection
    logger.info("Step 6: Final Model Selection and Testing")
    final_model_info = select_best_model(all_models, calibrated_model_info, datasets, logger)

    # Create final test plots
    create_final_test_plots(
        final_model_info['model'],
        datasets.get('X_test_scaled', datasets['X_test']),
        datasets['y_test'],
        plot_dir,
        logger
    )

    # Save final model
    import joblib
    final_model_path = os.path.join(model_dir, "final_model.pkl")
    with open(final_model_path, "wb") as f:
        pickle.dump(final_model_info, f)

    # Also save the model using joblib for production deployment
    prod_model_path = os.path.join(model_dir, "final_model_prod.joblib")
    joblib.dump(final_model_info['model'], prod_model_path)

    logger.info(f"Final model saved to: {final_model_path}")
    logger.info(f"Production model saved to: {prod_model_path}")

    # Final summary
    logger.info("\n=== MODEL DEVELOPMENT PIPELINE COMPLETE ===")
    logger.info(f"Selected Model: {final_model_info['model_name']}")
    logger.info("Test Performance:")
    for metric, value in final_model_info['test_metrics'].items():
        if metric not in ['confusion_matrix', 'classification_report']:
            logger.info(f"  {metric}: {value:.4f}")

    logger.info(f"All artifacts saved to:")
    logger.info(f"  - Logs: {log_dir}")
    logger.info(f"  - Plots: {plot_dir}")
    logger.info(f"  - Models: {model_dir}")

    return final_model_info


if __name__ == "__main__":
    run_pipeline()