#!/usr/bin/env python3
"""
Step 4 Execution Script
Simple script to run the complete Step 4 model training and evaluation pipeline
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the orchestrator
from code.step4_model_training.step4_orchestrator import main

if __name__ == "__main__":
    print("🤖 NBE Prediction Model - Step 4: Training & Evaluation")
    print("=" * 65)
    print()

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)

    # Run the main pipeline
    success = main()

    if success:
        print("\n✅ Step 4 completed successfully!")
        print("\n📋 Next Steps:")
        print("   1. Review generated plots in plots/step4_model_training/")
        print("   2. Check evaluation results in step4_evaluation_results_*.json")
        print("   3. Proceed to Step 6: API Development")
    else:
        print("\n❌ Step 4 failed. Check logs for details.")
        sys.exit(1)