"""Main data pipeline - orchestrates the complete data processing workflow"""

import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.data.preprocessor import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__, "logs/pipeline.log")


class DataPipeline:
    """Complete data processing pipeline for cancer prediction"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize DataPipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.loader = DataLoader(config_path)
        self.validator = DataValidator(config_path)
        self.preprocessor = DataPreprocessor(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        
        logger.info("="*80)
        logger.info("DATA PIPELINE INITIALIZED")
        logger.info("="*80)
    
    def run_pipeline(
        self,
        validate: bool = True,
        engineer_features: bool = True,
        save_processed: bool = True
    ) -> tuple:
        """
        Run complete data pipeline
        
        Args:
            validate: Whether to validate data
            engineer_features: Whether to apply feature engineering
            save_processed: Whether to save processed data
            
        Returns:
            Tuple of (X_train, X_test, X_val, y_train, y_test, y_val)
        """
        logger.info("Starting data pipeline...")
        
        # Step 1: Load data
        logger.info("\n" + "="*80)
        logger.info("STEP 1: LOADING DATA")
        logger.info("="*80)
        df = self.loader.load_data()
        logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Step 2: Validate data
        if validate:
            logger.info("\n" + "="*80)
            logger.info("STEP 2: VALIDATING DATA")
            logger.info("="*80)
            validation_report = self.validator.generate_validation_report(df)
            
            if not validation_report['overall_valid']:
                logger.warning("⚠️ Data validation found issues (continuing anyway)")
                for key, value in validation_report.items():
                    if value and key != 'overall_valid':
                        logger.warning(f"  {key}: {value}")
            else:
                logger.info("✅ Data validation passed")
        
        # Step 3: Feature Engineering (before preprocessing)
        if engineer_features:
            logger.info("\n" + "="*80)
            logger.info("STEP 3: FEATURE ENGINEERING")
            logger.info("="*80)
            df = self.feature_engineer.engineer_features(df)
            logger.info(f"Features after engineering: {df.shape[1]} columns")
        
        # Step 4: Split data
        logger.info("\n" + "="*80)
        logger.info("STEP 4: SPLITTING DATA")
        logger.info("="*80)
        train_df, test_df, val_df = self.loader.split_data(df)
        logger.info(f"Train: {train_df.shape[0]} | Test: {test_df.shape[0]} | Val: {val_df.shape[0] if val_df is not None else 0}")
        
        # Step 5: Preprocess data
        logger.info("\n" + "="*80)
        logger.info("STEP 5: PREPROCESSING DATA")
        logger.info("="*80)
        
        # Fit on training data
        train_processed = self.preprocessor.preprocess_pipeline(train_df, fit=True, include_target=True)
        
        # Transform test and validation data
        test_processed = self.preprocessor.preprocess_pipeline(test_df, fit=False, include_target=True)
        
        if val_df is not None:
            val_processed = self.preprocessor.preprocess_pipeline(val_df, fit=False, include_target=True)
        else:
            val_processed = None
        
        logger.info(f"Processed shapes - Train: {train_processed.shape}, Test: {test_processed.shape}")
        
        # Step 6: Separate features and target
        logger.info("\n" + "="*80)
        logger.info("STEP 6: SEPARATING FEATURES AND TARGET")
        logger.info("="*80)
        
        target_col = self.config.get('features.target')
        
        X_train = train_processed.drop(columns=[target_col])
        y_train = train_processed[target_col]
        
        X_test = test_processed.drop(columns=[target_col])
        y_test = test_processed[target_col]
        
        if val_processed is not None:
            X_val = val_processed.drop(columns=[target_col])
            y_val = val_processed[target_col]
        else:
            X_val = None
            y_val = None
        
        logger.info(f"Feature matrix shape: {X_train.shape}")
        logger.info(f"Target distribution in training: {y_train.value_counts().to_dict()}")
        
        # Step 7: Save processed data
        if save_processed:
            logger.info("\n" + "="*80)
            logger.info("STEP 7: SAVING PROCESSED DATA")
            logger.info("="*80)
            
            processed_dir = Path(self.config.get('paths.processed_data'))
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Save datasets
            self.loader.save_data(train_processed, f"{processed_dir}/train_processed.csv")
            self.loader.save_data(test_processed, f"{processed_dir}/test_processed.csv")
            if val_processed is not None:
                self.loader.save_data(val_processed, f"{processed_dir}/val_processed.csv")
            
            # Save preprocessors
            self.preprocessor.save_preprocessors()
            
            logger.info("✅ Processed data saved")
        
        # Step 8: Pipeline Summary
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Test samples: {X_test.shape[0]}")
        if X_val is not None:
            logger.info(f"Validation samples: {X_val.shape[0]}")
        logger.info(f"Total features: {X_train.shape[1]}")
        logger.info(f"Target: {target_col}")
        logger.info("="*80 + "\n")
        
        return X_train, X_test, X_val, y_train, y_test, y_val


def main():
    """Run the data pipeline"""
    
    print("\n" + "="*80)
    print("CANCER PREDICTION - DATA PIPELINE")
    print("="*80 + "\n")
    
    # Initialize and run pipeline
    pipeline = DataPipeline()
    
    X_train, X_test, X_val, y_train, y_test, y_val = pipeline.run_pipeline(
        validate=True,
        engineer_features=True,
        save_processed=True
    )
    
    # Display summary
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"\n✅ Data pipeline completed successfully!")
    print(f"\nDataset Splits:")
    print(f"  Training:   {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    print(f"  Test:       {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
    if X_val is not None:
        print(f"  Validation: {X_val.shape[0]:,} samples × {X_val.shape[1]} features")
    
    print(f"\nTarget Variable: Survival_Prediction")
    print(f"  Training distribution:")
    print(f"    Survived (1):     {(y_train == 1).sum():,} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")
    print(f"    Not Survived (0): {(y_train == 0).sum():,} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")
    
    print(f"\nProcessed Data Location:")
    print(f"  data/processed/train_processed.csv")
    print(f"  data/processed/test_processed.csv")
    if X_val is not None:
        print(f"  data/processed/val_processed.csv")
    
    print(f"\nPreprocessors saved to:")
    print(f"  models/artifacts/preprocessors.pkl")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("  1. Explore processed data in notebooks")
    print("  2. Train baseline models")
    print("  3. Run hyperparameter optimization")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
