"""Model training with MLflow tracking and model registry"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
import argparse
import json
from datetime import datetime
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score

from src.utils.logger import setup_logger
from src.utils.config import Config
from src.utils.helpers import save_model

logger = setup_logger(__name__, "logs/training.log")


class ModelTrainer:
    """Train and evaluate ML models with MLflow tracking"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = Config(config_path)
        self.models = {}
        self.results = {}
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config.get('mlflow.tracking_uri', 'mlruns'))
        mlflow.set_experiment(self.config.get('mlflow.experiment_name', 'cancer-prediction'))
        
        logger.info("ModelTrainer initialized")
    
    def load_processed_data(self):
        """Load processed train/test/val data"""
        logger.info("Loading processed data...")
        
        processed_dir = Path(self.config.get('paths.processed_data'))
        
        train_df = pd.read_csv(f"{processed_dir}/train_processed.csv")
        test_df = pd.read_csv(f"{processed_dir}/test_processed.csv")
        val_df = pd.read_csv(f"{processed_dir}/val_processed.csv")
        
        target_col = self.config.get('features.target')
        
        self.X_train = train_df.drop(columns=[target_col])
        self.y_train = train_df[target_col]
        
        self.X_test = test_df.drop(columns=[target_col])
        self.y_test = test_df[target_col]
        
        self.X_val = val_df.drop(columns=[target_col])
        self.y_val = val_df[target_col]
        
        logger.info(f"Data loaded - Train: {self.X_train.shape}, Test: {self.X_test.shape}, Val: {self.X_val.shape}")
    
    def initialize_models(self, quick_mode: bool = False):
        """Initialize model suite"""
        logger.info("Initializing models...")
        
        if quick_mode:
            # Fast models for CI/CD
            self.models = {
                'logistic_regression': LogisticRegression(max_iter=100, random_state=42),
                'random_forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
                'knn': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            }
        else:
            # Full model suite
            self.models = {
                'logistic_regression': LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    n_jobs=-1
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=self.config.get('models.random_forest.n_estimators', 200),
                    max_depth=self.config.get('models.random_forest.max_depth', 20),
                    min_samples_split=self.config.get('models.random_forest.min_samples_split', 10),
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=self.config.get('models.gradient_boosting.n_estimators', 200),
                    learning_rate=self.config.get('models.gradient_boosting.learning_rate', 0.1),
                    max_depth=self.config.get('models.gradient_boosting.max_depth', 5),
                    random_state=42
                ),
                'xgboost': XGBClassifier(
                    n_estimators=self.config.get('models.xgboost.n_estimators', 200),
                    learning_rate=self.config.get('models.xgboost.learning_rate', 0.1),
                    max_depth=self.config.get('models.xgboost.max_depth', 7),
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                ),
                'lightgbm': LGBMClassifier(
                    n_estimators=self.config.get('models.lightgbm.n_estimators', 200),
                    learning_rate=self.config.get('models.lightgbm.learning_rate', 0.1),
                    max_depth=self.config.get('models.lightgbm.max_depth', 7),
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                'svm': SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=42
                ),
                'knn': KNeighborsClassifier(
                    n_neighbors=self.config.get('models.knn.n_neighbors', 7),
                    n_jobs=-1
                )
            }
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def train_model(self, name: str, model, use_validation: bool = True):
        """Train a single model with MLflow tracking"""
        logger.info(f"\n{'='*80}\nTraining {name}\n{'='*80}")
        
        with mlflow.start_run(run_name=name):
            # Log parameters
            mlflow.log_params({
                'model_type': name,
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'num_features': self.X_train.shape[1]
            })
            
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            
            # Train
            start_time = datetime.now()
            model.fit(self.X_train, self.y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            mlflow.log_metric('training_time_seconds', training_time)
            logger.info(f"Training completed in {training_time:.2f}s")
            
            # Evaluate on validation set
            if use_validation:
                val_metrics = self._evaluate(model, self.X_val, self.y_val, 'validation')
                for metric_name, value in val_metrics.items():
                    mlflow.log_metric(f'val_{metric_name}', value)
            
            # Evaluate on test set
            test_metrics = self._evaluate(model, self.X_test, self.y_test, 'test')
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f'test_{metric_name}', value)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='roc_auc', n_jobs=-1)
            mlflow.log_metric('cv_auc_mean', cv_scores.mean())
            mlflow.log_metric('cv_auc_std', cv_scores.std())
            
            logger.info(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"cancer-prediction-{name}"
            )
            
            # Save locally
            model_dir = Path(self.config.get('paths.models'))
            model_dir.mkdir(parents=True, exist_ok=True)
            save_model(model, f"{model_dir}/{name}_model.pkl")
            
            # Store results
            self.results[name] = {
                'model': model,
                'test_metrics': test_metrics,
                'val_metrics': val_metrics if use_validation else None,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'training_time': training_time
            }
            
            logger.info(f"✅ {name} completed - Test AUC: {test_metrics['auc']:.4f}")
    
    def _evaluate(self, model, X, y, dataset_name: str) -> dict:
        """Evaluate model performance"""
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_pred_proba)
        }
        
        logger.info(f"{dataset_name.capitalize()} Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def train_all_models(self, quick_mode: bool = False):
        """Train all models in the suite"""
        logger.info("\n" + "="*80)
        logger.info("STARTING MODEL TRAINING SUITE")
        logger.info("="*80)
        
        self.load_processed_data()
        self.initialize_models(quick_mode)
        
        for name, model in self.models.items():
            try:
                self.train_model(name, model)
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING SUMMARY")
        logger.info("="*80)
        
        results_df = pd.DataFrame({
            name: {
                'Test Accuracy': res['test_metrics']['accuracy'],
                'Test AUC': res['test_metrics']['auc'],
                'Test F1': res['test_metrics']['f1'],
                'CV AUC': res['cv_auc_mean'],
                'Training Time (s)': res['training_time']
            }
            for name, res in self.results.items()
        }).T
        
        results_df = results_df.sort_values('Test AUC', ascending=False)
        logger.info("\n" + results_df.to_string())
        
        # Save results
        results_path = Path("models/training_results.csv")
        results_df.to_csv(results_path)
        logger.info(f"\nResults saved to {results_path}")
        
        return results_df
    
    def select_best_model(self, metric: str = 'auc') -> tuple:
        """Select best model based on metric"""
        best_name = max(
            self.results.keys(),
            key=lambda x: self.results[x]['test_metrics'][metric]
        )
        best_model = self.results[best_name]['model']
        best_score = self.results[best_name]['test_metrics'][metric]
        
        logger.info(f"\n🏆 Best Model: {best_name} (Test {metric.upper()}: {best_score:.4f})")
        
        # Register in MLflow Model Registry
        with mlflow.start_run(run_name=f"best_model_{best_name}"):
            mlflow.log_param('selected_as_best', True)
            mlflow.log_param('selection_metric', metric)
            
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, "cancer-prediction-production")
        
        return best_name, best_model, best_score


def main():
    parser = argparse.ArgumentParser(description='Train cancer prediction models')
    parser.add_argument('--quick-validation', action='store_true',
                       help='Quick training for CI/CD validation')
    parser.add_argument('--metric', type=str, default='auc',
                       choices=['accuracy', 'auc', 'f1', 'precision', 'recall'],
                       help='Metric for model selection')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    
    # Train all models
    results = trainer.train_all_models(quick_mode=args.quick_validation)
    
    # Select best model
    best_name, best_model, best_score = trainer.select_best_model(metric=args.metric)
    
    # Save candidate metrics for comparison
    candidate_metrics = {
        'model_name': best_name,
        'accuracy': trainer.results[best_name]['test_metrics']['accuracy'],
        'auc': trainer.results[best_name]['test_metrics']['auc'],
        'f1': trainer.results[best_name]['test_metrics']['f1'],
        'precision': trainer.results[best_name]['test_metrics']['precision'],
        'recall': trainer.results[best_name]['test_metrics']['recall'],
        'timestamp': datetime.now().isoformat()
    }
    
    Path('models/candidate').mkdir(parents=True, exist_ok=True)
    with open('models/candidate/metrics.json', 'w') as f:
        json.dump(candidate_metrics, f, indent=2)
    
    print(f"\n✅ Training complete! Best model: {best_name} ({args.metric.upper()}: {best_score:.4f})")


if __name__ == "__main__":
    main()
