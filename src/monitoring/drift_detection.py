"""Drift detection for features and model performance"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
import json
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from src.utils.config import Config
from src.utils.helpers import load_model
import joblib

logger = setup_logger(__name__, "logs/drift_detection.log")


class DriftDetector:
    """Detect feature drift and model performance degradation"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = Config(config_path)
        self.reference_data = None
        self.current_data = None
        self.drift_threshold = self.config.get('monitoring.drift_threshold', 0.05)
        self.performance_threshold = self.config.get('monitoring.performance_threshold', 0.02)
        
        logger.info("DriftDetector initialized")
    
    def load_reference_data(self):
        """Load reference (training) data"""
        logger.info("Loading reference data...")
        
        reference_path = Path(self.config.get('paths.processed_data')) / "train_processed.csv"
        self.reference_data = pd.read_csv(reference_path)
        
        logger.info(f"Reference data loaded: {self.reference_data.shape}")
    
    def load_current_data(self, path: str = None):
        """Load current/production data"""
        if path is None:
            # In production, load from database or recent predictions
            # For demo, use validation set
            path = Path(self.config.get('paths.processed_data')) / "val_processed.csv"
        
        logger.info(f"Loading current data from {path}...")
        self.current_data = pd.read_csv(path)
        
        logger.info(f"Current data loaded: {self.current_data.shape}")
    
    def detect_numerical_drift(self, feature: str) -> dict:
        """Detect drift in numerical feature using KS test"""
        ref_values = self.reference_data[feature].dropna()
        curr_values = self.current_data[feature].dropna()
        
        # Kolmogorov-Smirnov test
        statistic, p_value = ks_2samp(ref_values, curr_values)
        
        drift_detected = p_value < self.drift_threshold
        
        return {
            'feature': feature,
            'type': 'numerical',
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': drift_detected,
            'ref_mean': float(ref_values.mean()),
            'curr_mean': float(curr_values.mean()),
            'mean_change_pct': float((curr_values.mean() - ref_values.mean()) / ref_values.mean() * 100)
        }
    
    def detect_categorical_drift(self, feature: str) -> dict:
        """Detect drift in categorical feature using chi-square test"""
        ref_dist = self.reference_data[feature].value_counts(normalize=True)
        curr_dist = self.current_data[feature].value_counts(normalize=True)
        
        # Align categories
        all_categories = set(ref_dist.index) | set(curr_dist.index)
        ref_counts = [self.reference_data[feature].value_counts().get(cat, 0) for cat in all_categories]
        curr_counts = [self.current_data[feature].value_counts().get(cat, 0) for cat in all_categories]
        
        # Chi-square test
        try:
            statistic, p_value, _, _ = chi2_contingency([ref_counts, curr_counts])
            drift_detected = p_value < self.drift_threshold
        except:
            statistic, p_value = 0, 1
            drift_detected = False
        
        return {
            'feature': feature,
            'type': 'categorical',
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': drift_detected,
            'ref_distribution': ref_dist.to_dict(),
            'curr_distribution': curr_dist.to_dict()
        }
    
    def detect_feature_drift(self) -> dict:
        """Detect drift across all features"""
        logger.info("Detecting feature drift...")
        
        self.load_reference_data()
        self.load_current_data()
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'features': {},
            'drift_detected_count': 0,
            'total_features': 0
        }
        
        target_col = self.config.get('features.target')
        
        for feature in self.reference_data.columns:
            if feature == target_col:
                continue
            
            try:
                if self.reference_data[feature].dtype in ['int64', 'float64']:
                    result = self.detect_numerical_drift(feature)
                else:
                    result = self.detect_categorical_drift(feature)
                
                drift_results['features'][feature] = result
                
                if result['drift_detected']:
                    drift_results['drift_detected_count'] += 1
                    logger.warning(f"⚠️  Drift detected in {feature}: p-value={result['p_value']:.4f}")
                
                drift_results['total_features'] += 1
                
            except Exception as e:
                logger.error(f"Error detecting drift for {feature}: {str(e)}")
                continue
        
        drift_results['drift_percentage'] = (
            drift_results['drift_detected_count'] / drift_results['total_features'] * 100
        )
        
        logger.info(f"Drift detection complete: {drift_results['drift_detected_count']}/{drift_results['total_features']} features drifted ({drift_results['drift_percentage']:.1f}%)")
        
        return drift_results
    
    def detect_performance_drift(self) -> dict:
        """Detect model performance degradation"""
        logger.info("Detecting performance drift...")
        
        self.load_current_data()
        
        target_col = self.config.get('features.target')
        X_curr = self.current_data.drop(columns=[target_col])
        y_curr = self.current_data[target_col]
        
        # Load model
        model_path = Path(self.config.get('paths.models')) / "best_model.pkl"
        if not model_path.exists():
            model_files = list(Path(self.config.get('paths.models')).glob("*_model.pkl"))
            model_path = model_files[0] if model_files else None
        
        if model_path is None:
            logger.error("No model found for performance drift detection")
            return {'error': 'No model found'}
        
        model = load_model(str(model_path))
        
        # Predict
        y_pred = model.predict(X_curr)
        y_pred_proba = model.predict_proba(X_curr)[:, 1]
        
        # Calculate metrics
        current_metrics = {
            'accuracy': accuracy_score(y_curr, y_pred),
            'auc': roc_auc_score(y_curr, y_pred_proba),
            'f1': f1_score(y_curr, y_pred)
        }
        
        # Load baseline metrics
        baseline_path = Path("models/baseline/metrics.json")
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                baseline_metrics = json.load(f)
        else:
            # Use current as baseline
            baseline_metrics = current_metrics
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            with open(baseline_path, 'w') as f:
                json.dump(baseline_metrics, f, indent=2)
        
        # Calculate degradation
        performance_drift = {}
        degradation_detected = False
        
        for metric in ['accuracy', 'auc', 'f1']:
            baseline_value = baseline_metrics.get(metric, current_metrics[metric])
            current_value = current_metrics[metric]
            degradation = baseline_value - current_value
            degradation_pct = (degradation / baseline_value * 100) if baseline_value > 0 else 0
            
            performance_drift[metric] = {
                'baseline': baseline_value,
                'current': current_value,
                'degradation': degradation,
                'degradation_pct': degradation_pct,
                'threshold_exceeded': degradation > self.performance_threshold
            }
            
            if degradation > self.performance_threshold:
                degradation_detected = True
                logger.warning(f"⚠️  Performance degradation in {metric}: {degradation:.4f} ({degradation_pct:.1f}%)")
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(model_path),
            'metrics': performance_drift,
            'degradation_detected': degradation_detected,
            'retraining_recommended': degradation_detected
        }
        
        logger.info(f"Performance drift detection complete: {'Degradation detected' if degradation_detected else 'No degradation'}")
        
        return result
    
    def detect_drift(self) -> dict:
        """Run complete drift detection pipeline"""
        logger.info("\n" + "="*80)
        logger.info("DRIFT DETECTION PIPELINE")
        logger.info("="*80)
        
        # Feature drift
        feature_drift_results = self.detect_feature_drift()
        
        # Performance drift
        performance_drift_results = self.detect_performance_drift()
        
        # Combined results
        results = {
            'timestamp': datetime.now().isoformat(),
            'feature_drift': feature_drift_results,
            'performance_drift': performance_drift_results,
            'overall_drift_detected': (
                feature_drift_results['drift_percentage'] > 10 or
                performance_drift_results.get('degradation_detected', False)
            ),
            'retraining_triggered': False
        }
        
        # Save report
        report_path = Path("logs/drift_reports")
        report_path.mkdir(parents=True, exist_ok=True)
        
        report_file = report_path / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Drift report saved to {report_file}")
        
        # Trigger retraining if needed
        if results['overall_drift_detected']:
            logger.warning("🚨 DRIFT DETECTED - Retraining recommended")
            results['retraining_triggered'] = True
            # In production, trigger CT pipeline here
        else:
            logger.info("✅ No significant drift detected")
        
        logger.info("="*80 + "\n")
        
        return results


def main():
    """Run drift detection"""
    detector = DriftDetector()
    results = detector.detect_drift()
    
    print(f"\n{'='*80}")
    print("DRIFT DETECTION SUMMARY")
    print(f"{'='*80}")
    print(f"\nFeature Drift: {results['feature_drift']['drift_detected_count']}/{results['feature_drift']['total_features']} features ({results['feature_drift']['drift_percentage']:.1f}%)")
    print(f"\nPerformance Drift: {'Yes' if results['performance_drift'].get('degradation_detected') else 'No'}")
    
    if results['performance_drift'].get('metrics'):
        print("\nMetrics:")
        for metric, data in results['performance_drift']['metrics'].items():
            print(f"  {metric.upper()}: {data['baseline']:.4f} → {data['current']:.4f} ({data['degradation_pct']:+.1f}%)")
    
    print(f"\nRetraining Recommended: {'Yes' if results['overall_drift_detected'] else 'No'}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
