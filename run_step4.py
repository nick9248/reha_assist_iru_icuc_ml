#!/usr/bin/env python3
"""
Simple Step 4 Execution Script
Standalone script to run Step 4 without external dependencies
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime


def setup_project_paths():
    """Setup project paths"""
    # Get project root (assuming script is in project root)
    project_root = Path(__file__).parent.absolute()

    # Setup paths
    paths = {
        'project_root': project_root,
        'data_path': project_root / 'data',
        'logs_path': project_root / 'logs',
        'plots_path': project_root / 'plots',
        'models_path': project_root / 'models'
    }

    # Create directories if they don't exist
    for path_name, path in paths.items():
        if path_name != 'project_root':
            path.mkdir(parents=True, exist_ok=True)

    return paths


def setup_simple_logger():
    """Setup simple console logger"""
    logger = logging.getLogger('Step4Simple')
    logger.setLevel(logging.INFO)

    # Console handler
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def check_step2_files(data_path, logger):
    """Check if Step 2 files exist"""
    processed_dir = data_path / 'processed'

    required_patterns = [
        'step2_baseline_train_*.csv',
        'step2_baseline_test_*.csv',
        'step2_enhanced_train_*.csv',
        'step2_enhanced_test_*.csv'
    ]

    missing = []
    for pattern in required_patterns:
        files = list(processed_dir.glob(pattern))
        if not files:
            missing.append(pattern)

    if missing:
        logger.error(f"Missing Step 2 files: {missing}")
        logger.error("Please run Step 2 preprocessing first")
        return False

    logger.info("✅ All Step 2 files found")
    return True


def main():
    """Main execution function"""
    print("🤖 NBE Prediction - Step 4: Model Training & Evaluation")
    print("=" * 60)

    # Setup
    paths = setup_project_paths()
    logger = setup_simple_logger()

    logger.info("Starting Step 4 execution")
    logger.info(f"Project root: {paths['project_root']}")

    try:
        # Add project root to Python path
        sys.path.insert(0, str(paths['project_root']))

        # Check Step 2 outputs
        if not check_step2_files(paths['data_path'], logger):
            return False

        # Import modules (after path setup)
        from code.step4_model_training.model_trainer import ModelTrainer
        from code.step4_model_training.model_evaluator import ModelEvaluator

        # Initialize trainer and evaluator
        logger.info("🔧 Initializing trainer and evaluator...")
        trainer = ModelTrainer(paths['models_path'], paths['logs_path'])
        evaluator = ModelEvaluator(paths['plots_path'], paths['logs_path'])

        # Train all models
        logger.info("🚀 Training models...")
        baseline_models, enhanced_models, saved_files = trainer.train_all_models(paths['data_path'])

        logger.info(f"✅ Training completed: {len(baseline_models)} baseline + {len(enhanced_models)} enhanced models")

        # Load test datasets for evaluation
        logger.info("📊 Loading test datasets...")
        test_datasets = trainer.load_training_data(paths['data_path'])

        # Run comprehensive evaluation
        logger.info("🔍 Evaluating models...")
        evaluation_results = evaluator.generate_comprehensive_evaluation(
            baseline_models, enhanced_models, test_datasets
        )

        # Print summary
        print("\n" + "=" * 60)
        print("🏆 RESULTS SUMMARY")
        print("=" * 60)

        comparison = evaluation_results['comparison_results']

        # Best models
        if comparison.get('best_models'):
            best_baseline = comparison['best_models'].get('baseline')
            best_enhanced = comparison['best_models'].get('enhanced')

            print("\n📊 BEST PERFORMING MODELS:")
            if best_baseline:
                print(f"   Baseline: {best_baseline['model']} (AUC: {best_baseline['auc_roc']:.4f})")
            if best_enhanced:
                print(f"   Enhanced: {best_enhanced['model']} (AUC: {best_enhanced['auc_roc']:.4f})")

        # Show improvement
        if comparison.get('improvement_analysis'):
            print("\n📈 IMPROVEMENT WITH ENHANCED FEATURES:")
            for model, analysis in comparison['improvement_analysis'].items():
                improvement_pct = analysis['improvement_percentage']
                print(f"   {model}: {improvement_pct:+.2f}%")

        # Recommendations
        if comparison.get('recommendations'):
            print("\n💡 RECOMMENDATIONS:")
            for rec in comparison['recommendations']:
                print(f"   • {rec}")

        # Generated files
        print("\n📁 GENERATED FILES:")
        for plot_type, path in evaluation_results['plot_paths'].items():
            print(f"   📊 {plot_type}: {Path(path).name}")

        results_file = Path(evaluation_results['results_file']).name
        print(f"   📋 Complete results: {results_file}")

        print("\n✅ Step 4 completed successfully!")
        print("\n📋 Next Steps:")
        print("   1. Review plots in plots/step4_model_training/")
        print("   2. Check detailed results in step4_evaluation_results_*.json")
        print("   3. Proceed to Step 6: API Development")

        return True

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you're running from the project root directory")
        return False

    except Exception as e:
        logger.error(f"Error in Step 4 execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Step 4 failed. Check error messages above.")
        sys.exit(1)