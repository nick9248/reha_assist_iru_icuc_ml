"""
Step 4 Orchestrator: Main execution script for model training and evaluation
Coordinates the complete training and evaluation pipeline
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import Step 4 modules
from code.step4_model_training.model_trainer import ModelTrainer
from code.step4_model_training.model_evaluator import ModelEvaluator

def load_environment_paths():
    """Load paths from environment or use defaults"""
    try:
        from dotenv import load_dotenv
        import os

        load_dotenv()

        project_root = Path(os.getenv('PROJECT_ROOT', Path(__file__).parent.parent))
        data_path = project_root / 'data'
        logs_path = project_root / 'logs'
        plots_path = project_root / 'plots'
        models_path = project_root / 'models'

    except ImportError:
        # Fallback if python-dotenv not available
        project_root = Path(__file__).parent.parent
        data_path = project_root / 'data'
        logs_path = project_root / 'logs'
        plots_path = project_root / 'plots'
        models_path = project_root / 'models'

    # Create directories if they don't exist
    for path in [data_path, logs_path, plots_path, models_path]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        'project_root': project_root,
        'data_path': data_path,
        'logs_path': logs_path,
        'plots_path': plots_path,
        'models_path': models_path
    }

def setup_orchestrator_logger(log_path: Path) -> logging.Logger:
    """Setup logger for orchestrator"""
    logger = logging.getLogger('Step4Orchestrator')
    logger.setLevel(logging.INFO)

    # Create log directory
    log_dir = log_path / 'step4'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create file handler with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'step4_orchestrator_{timestamp}.log'

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def validate_step2_outputs(data_path: Path, logger: logging.Logger) -> bool:
    """
    Validate that Step 2 outputs are available

    Args:
        data_path: Path to data directory
        logger: Logger instance

    Returns:
        bool: True if all required files are present
    """
    logger.info("Validating Step 2 outputs")

    processed_dir = data_path / 'processed'

    required_files = [
        'step2_baseline_train_*.csv',
        'step2_baseline_test_*.csv',
        'step2_enhanced_train_*.csv',
        'step2_enhanced_test_*.csv'
    ]

    missing_files = []

    for file_pattern in required_files:
        matching_files = list(processed_dir.glob(file_pattern))
        if not matching_files:
            missing_files.append(file_pattern)

    if missing_files:
        logger.error(f"Missing Step 2 output files: {missing_files}")
        logger.error("Please run Step 2 data preprocessing first")
        return False

    logger.info("All Step 2 output files found")
    return True

def run_training_pipeline(paths: dict, logger: logging.Logger) -> tuple:
    """
    Execute the complete training pipeline

    Args:
        paths: Dictionary of project paths
        logger: Logger instance

    Returns:
        tuple: (baseline_models, enhanced_models, saved_files)
    """
    logger.info("Starting model training pipeline")

    # Initialize trainer
    trainer = ModelTrainer(paths['models_path'], paths['logs_path'])

    # Train all models
    baseline_models, enhanced_models, saved_files = trainer.train_all_models(paths['data_path'])

    logger.info("Training pipeline completed successfully")
    return baseline_models, enhanced_models, saved_files

def run_evaluation_pipeline(baseline_models: dict, enhanced_models: dict,
                          paths: dict, logger: logging.Logger) -> dict:
    """
    Execute the complete evaluation pipeline

    Args:
        baseline_models: Trained baseline models
        enhanced_models: Trained enhanced models
        paths: Dictionary of project paths
        logger: Logger instance

    Returns:
        dict: Complete evaluation results
    """
    logger.info("Starting model evaluation pipeline")

    # Initialize evaluator
    evaluator = ModelEvaluator(paths['plots_path'], paths['logs_path'])

    # Load test datasets
    trainer = ModelTrainer(paths['models_path'], paths['logs_path'])
    test_datasets = trainer.load_training_data(paths['data_path'])

    # Run comprehensive evaluation
    evaluation_results = evaluator.generate_comprehensive_evaluation(
        baseline_models, enhanced_models, test_datasets
    )

    logger.info("Evaluation pipeline completed successfully")
    return evaluation_results

def print_results_summary(evaluation_results: dict, logger: logging.Logger):
    """
    Print summary of results to console and log

    Args:
        evaluation_results: Complete evaluation results
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("STEP 4 RESULTS SUMMARY")
    logger.info("=" * 60)

    comparison = evaluation_results['comparison_results']

    # Best models
    if comparison.get('best_models'):
        best_baseline = comparison['best_models'].get('baseline')
        best_enhanced = comparison['best_models'].get('enhanced')

        logger.info("\n🏆 BEST PERFORMING MODELS:")
        if best_baseline:
            logger.info(f"   Baseline: {best_baseline['model']} (AUC: {best_baseline['auc_roc']:.4f})")
        if best_enhanced:
            logger.info(f"   Enhanced: {best_enhanced['model']} (AUC: {best_enhanced['auc_roc']:.4f})")

    # Improvement analysis
    if comparison.get('improvement_analysis'):
        logger.info("\n📈 IMPROVEMENT ANALYSIS:")
        for model, analysis in comparison['improvement_analysis'].items():
            improvement_pct = analysis['improvement_percentage']
            logger.info(f"   {model}: {improvement_pct:+.2f}% improvement with enhanced features")

    # Recommendations
    if comparison.get('recommendations'):
        logger.info("\n💡 RECOMMENDATIONS:")
        for rec in comparison['recommendations']:
            logger.info(f"   • {rec}")

    # Generated artifacts
    logger.info("\n📁 GENERATED ARTIFACTS:")
    for plot_type, path in evaluation_results['plot_paths'].items():
        logger.info(f"   • {plot_type}: {Path(path).name}")

    logger.info(f"\n📊 Complete results saved: {Path(evaluation_results['results_file']).name}")
    logger.info("=" * 60)

def main():
    """Main execution function"""
    print("🚀 Starting Step 4: Model Training & Evaluation")
    print("=" * 60)

    # Load environment and setup paths
    paths = load_environment_paths()
    logger = setup_orchestrator_logger(paths['logs_path'])

    logger.info("Step 4: Model Training & Evaluation started")
    logger.info(f"Project root: {paths['project_root']}")

    try:
        # Validate Step 2 outputs
        if not validate_step2_outputs(paths['data_path'], logger):
            print("❌ Step 2 outputs not found. Please run Step 2 first.")
            return False

        # Run training pipeline
        print("\n🔧 Training Models...")
        baseline_models, enhanced_models, training_files = run_training_pipeline(paths, logger)

        print(f"✅ Trained {len(baseline_models)} baseline and {len(enhanced_models)} enhanced models")

        # Run evaluation pipeline
        print("\n📊 Evaluating Models...")
        evaluation_results = run_evaluation_pipeline(baseline_models, enhanced_models, paths, logger)

        print("✅ Model evaluation completed")

        # Print results summary
        print_results_summary(evaluation_results, logger)

        print("\n🎉 Step 4 completed successfully!")
        print("Next: Review results and proceed to Step 6 (API Development)")

        logger.info("Step 4 orchestrator completed successfully")
        return True

    except Exception as e:
        error_msg = f"Error in Step 4 execution: {str(e)}"
        logger.error(error_msg)
        print(f"\n❌ {error_msg}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)