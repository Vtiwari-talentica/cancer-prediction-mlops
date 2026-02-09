"""Continuous Training (CT) Pipeline - Triggered on drift detection"""

import pandas as pd
from pathlib import Path
import sys
import mlflow
from datetime import datetime
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.pipeline import DataPipeline
from src.models.train_models import ModelTrainer
from src.monitoring.drift_detection import DriftDetector
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__, "logs/continuous_training.log")


class ContinuousTrainingPipeline:
    """Automated retraining pipeline triggered by drift"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = Config(config_path)
        self.ct_enabled = self.config.get('continuous_training.enabled', True)
        self.auto_deploy = self.config.get('continuous_training.auto_deploy', False)
        
        logger.info("Continuous Training Pipeline initialized")
    
    def should_retrain(self) -> tuple:
        """Check if retraining is needed"""
        logger.info("Checking retraining criteria...")
        
        # Run drift detection
        detector = DriftDetector()
        drift_results = detector.detect_drift()
        
        should_retrain = drift_results['overall_drift_detected']
        reason = []
        
        if drift_results['feature_drift']['drift_percentage'] > 10:
            reason.append(f"Feature drift: {drift_results['feature_drift']['drift_percentage']:.1f}%")
        
        if drift_results['performance_drift'].get('degradation_detected'):
            reason.append("Performance degradation detected")
        
        logger.info(f"Retraining needed: {should_retrain}")
        if reason:
            logger.info(f"Reasons: {', '.join(reason)}")
        
        return should_retrain, reason, drift_results
    
    def run_pipeline(self):
        """Run complete CT pipeline"""
        logger.info("\n" + "="*80)
        logger.info("CONTINUOUS TRAINING PIPELINE - START")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        with mlflow.start_run(run_name=f"ct_pipeline_{start_time.strftime('%Y%m%d_%H%M%S')}"):
            # Log pipeline metadata
            mlflow.log_param('pipeline_type', 'continuous_training')
            mlflow.log_param('triggered_at', start_time.isoformat())
            
            # Step 1: Check retraining criteria
            logger.info("\n" + "="*80)
            logger.info("STEP 1: DRIFT DETECTION")
            logger.info("="*80)
            
            should_retrain, reasons, drift_results = self.should_retrain()
            mlflow.log_param('retraining_needed', should_retrain)
            mlflow.log_param('drift_reasons', ', '.join(reasons) if reasons else 'none')
            
            if not should_retrain:
                logger.info("✅ No retraining needed - drift within acceptable limits")
                logger.info("="*80 + "\n")
                return {
                    'status': 'skipped',
                    'reason': 'No drift detected',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Step 2: Data Ingestion & Validation
            logger.info("\n" + "="*80)
            logger.info("STEP 2: DATA INGESTION & VALIDATION")
            logger.info("="*80)
            
            # In production, fetch fresh data from database/data lake
            # For now, we'll use existing data
            logger.info("Using existing validated data")
            
            # Step 3: Feature Engineering & Validation
            logger.info("\n" + "="*80)
            logger.info("STEP 3: FEATURE ENGINEERING & VALIDATION")
            logger.info("="*80)
            
            data_pipeline = DataPipeline()
            X_train, X_test, X_val, y_train, y_test, y_val = data_pipeline.run_pipeline(
                validate=True,
                engineer_features=True,
                save_processed=True
            )
            
            mlflow.log_param('train_samples', len(X_train))
            mlflow.log_param('test_samples', len(X_test))
            mlflow.log_param('num_features', X_train.shape[1])
            
            # Step 4: Model Training
            logger.info("\n" + "="*80)
            logger.info("STEP 4: MODEL TRAINING")
            logger.info("="*80)
            
            trainer = ModelTrainer()
            trainer.load_processed_data()
            trainer.initialize_models(quick_mode=False)
            
            results = trainer.train_all_models()
            
            # Step 5: Model Selection
            logger.info("\n" + "="*80)
            logger.info("STEP 5: MODEL SELECTION")
            logger.info("="*80)
            
            best_name, best_model, best_score = trainer.select_best_model(metric='auc')
            
            mlflow.log_param('best_model', best_name)
            mlflow.log_metric('best_auc', best_score)
            
            # Step 6: Model Registry
            logger.info("\n" + "="*80)
            logger.info("STEP 6: MLflow MODEL REGISTRY")
            logger.info("="*80)
            
            # Register new model version
            model_uri = mlflow.get_artifact_uri("model")
            model_details = mlflow.register_model(
                model_uri,
                "cancer-prediction-production"
            )
            
            logger.info(f"Model registered: {model_details.name} version {model_details.version}")
            
            # Step 7: Model Deployment Decision
            logger.info("\n" + "="*80)
            logger.info("STEP 7: DEPLOYMENT DECISION")
            logger.info("="*80)
            
            if self.auto_deploy:
                logger.info("Auto-deployment enabled - deploying new model...")
                self._deploy_model(best_model, best_name)
                deployment_status = 'deployed'
            else:
                logger.info("Auto-deployment disabled - model ready for manual deployment")
                deployment_status = 'staged'
            
            # Step 8: Update Baseline
            logger.info("\n" + "="*80)
            logger.info("STEP 8: UPDATE BASELINE METRICS")
            logger.info("="*80)
            
            new_baseline = {
                'model_name': best_name,
                'accuracy': trainer.results[best_name]['test_metrics']['accuracy'],
                'auc': trainer.results[best_name]['test_metrics']['auc'],
                'f1': trainer.results[best_name]['test_metrics']['f1'],
                'precision': trainer.results[best_name]['test_metrics']['precision'],
                'recall': trainer.results[best_name]['test_metrics']['recall'],
                'timestamp': datetime.now().isoformat(),
                'retrained_due_to': reasons
            }
            
            baseline_path = Path("models/baseline/metrics.json")
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            with open(baseline_path, 'w') as f:
                json.dump(new_baseline, f, indent=2)
            
            logger.info("Baseline metrics updated")
            
            # Summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            mlflow.log_metric('pipeline_duration_seconds', duration)
            
            logger.info("\n" + "="*80)
            logger.info("CONTINUOUS TRAINING PIPELINE - COMPLETE")
            logger.info("="*80)
            logger.info(f"Duration: {duration:.1f}s")
            logger.info(f"Best Model: {best_name} (AUC: {best_score:.4f})")
            logger.info(f"Deployment Status: {deployment_status}")
            logger.info("="*80 + "\n")
            
            return {
                'status': 'success',
                'best_model': best_name,
                'best_auc': best_score,
                'deployment_status': deployment_status,
                'duration_seconds': duration,
                'timestamp': end_time.isoformat()
            }
    
    def _deploy_model(self, model, model_name: str):
        """Deploy model to production"""
        from src.utils.helpers import save_model
        
        # Save as production model
        model_path = Path(self.config.get('paths.models')) / "best_model.pkl"
        save_model(model, str(model_path))
        
        logger.info(f"Model deployed to {model_path}")
        
        # In production: trigger Docker container restart or update model service
        logger.info("Model deployment complete - restart API services to use new model")


def main():
    """Run CT pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous Training Pipeline')
    parser.add_argument('--force', action='store_true', help='Force retraining regardless of drift')
    args = parser.parse_args()
    
    pipeline = ContinuousTrainingPipeline()
    
    if args.force:
        logger.info("Force retraining enabled - skipping drift check")
        result = pipeline.run_pipeline()
    else:
        result = pipeline.run_pipeline()
    
    print(f"\n{'='*80}")
    print("CT PIPELINE RESULT")
    print(f"{'='*80}")
    print(json.dumps(result, indent=2))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
